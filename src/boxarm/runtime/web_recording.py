from __future__ import annotations

"""Captura el renderer real del navegador para grabar ISO y dashboard."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from boxarm.config import CameraConfig, DashboardRecordingConfig, RecordingConfig
from boxarm.runtime.recording import VideoRecorder

logger = logging.getLogger(__name__)


def browser_base_url(capture_host: str, port: int) -> str:
    """Forma la URL desde el destino explicito configurado para Chromium."""
    host = capture_host.strip()
    if host in ("", "0.0.0.0", "::", "[::]"):
        raise ValueError(f"capture_host={capture_host!r} no es navegable")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def recording_page_url(base_url: str, cam_id: int,
                       dashboard: DashboardRecordingConfig) -> str:
    """URL privada del Chromium con su angulo de grabacion configurado."""
    return (f"{base_url}/cam/{cam_id}?az={dashboard.azimuth_deg}"
            f"&el={dashboard.elevation_deg}")


def crop_css_box(
    frame: np.ndarray,
    box: dict[str, float],
    viewport_width: int,
    viewport_height: int,
) -> np.ndarray:
    """Recorta un bounding-box CSS incluso si el screenshot tiene otro DPR."""
    height, width = frame.shape[:2]
    scale_x = width / viewport_width
    scale_y = height / viewport_height
    x1 = max(0, min(width, round(box["x"] * scale_x)))
    y1 = max(0, min(height, round(box["y"] * scale_y)))
    x2 = max(x1, min(width, round((box["x"] + box["width"]) * scale_x)))
    y2 = max(y1, min(height, round((box["y"] + box["height"]) * scale_y)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"panel ISO sin area visible: {box}")
    return frame[y1:y2, x1:x2].copy()


@dataclass
class _Capture:
    cam: CameraConfig
    page: Any
    iso_box: dict[str, float]
    snapshot_url: str
    iso_recorder: VideoRecorder
    dashboard_recorder: VideoRecorder


def _open_camera_page(browser: Any, cam: CameraConfig, cfg: RecordingConfig,
                      base_url: str, stop: Any) -> _Capture | None:
    viewport = {"width": cfg.dashboard.width, "height": cfg.dashboard.height}
    page = browser.new_page(viewport=viewport, device_scale_factor=1)
    url = recording_page_url(base_url, cam.id, cfg.dashboard)
    snapshot_url = f"{base_url}/cam/{cam.id}/snapshot"
    deadline = time.monotonic() + cfg.dashboard.startup_timeout_s

    while not stop.is_set() and time.monotonic() < deadline:
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=3000)
            if response is not None and response.ok:
                page.wait_for_selector("#iso", state="visible", timeout=3000)
                # Cancela el MJPEG infinito solo en este Chromium interno.
                # Un recurso JPEG finito permite que screenshot() termine.
                page.locator("#camera-stream").evaluate(
                    "(img, url) => { img.src = url + '?t=' + Date.now(); }",
                    snapshot_url,
                )
                page.wait_for_function(
                    "() => { const img = document.getElementById('camera-stream'); "
                    "return img && img.complete && img.naturalWidth > 0; }",
                    timeout=3000,
                )
                iso_box = page.locator("#lienzo").bounding_box()
                if iso_box is not None:
                    logger.info("[%s] capturador web listo en %s", cam.tag, url)
                    return _Capture(
                        cam=cam,
                        page=page,
                        iso_box=iso_box,
                        snapshot_url=snapshot_url,
                        iso_recorder=VideoRecorder(cfg, cam.id, "iso", cam.tag),
                        dashboard_recorder=VideoRecorder(cfg, cam.id, "dashboard", cam.tag),
                    )
        except Exception as exc:  # Playwright expone varias subclases segun el fallo
            logger.debug("[%s] panel web aun no disponible: %s", cam.tag, exc)
        stop.wait(min(0.5, max(0.0, deadline - time.monotonic())))

    page.close()
    if not stop.is_set():
        logger.error("[%s] el panel web no estuvo listo en %.1fs; no se grabara ISO/dashboard",
                     cam.tag, cfg.dashboard.startup_timeout_s)
    return None


def record_web_views(cameras: tuple[CameraConfig, ...], cfg: RecordingConfig,
                     port: int, stop: Any) -> None:
    """Graba el ISO web y/o el dashboard completo con un Chromium headless.

    Se toma UN screenshot por camara y ciclo. El ISO se obtiene recortando
    `#lienzo`, por lo que ambos archivos corresponden al mismo instante.
    """
    if not (cfg.type_enabled("iso") or cfg.type_enabled("dashboard")):
        return

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("recording.types.iso/dashboard requiere Playwright; "
                     "instale dependencias y ejecute `python -m playwright install chromium`")
        return

    captures: list[_Capture] = []
    base_url = browser_base_url(cfg.dashboard.capture_host, port)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            try:
                for cam in cameras:
                    capture = _open_camera_page(browser, cam, cfg, base_url, stop)
                    if capture is not None:
                        captures.append(capture)

                period = 1.0 / cfg.fps
                while captures and not stop.is_set():
                    started = time.monotonic()
                    for capture in captures:
                        try:
                            capture.page.locator("#camera-stream").evaluate(
                                "(img, url) => new Promise(resolve => { "
                                "img.onload = () => resolve(true); "
                                "img.onerror = () => resolve(false); "
                                "img.src = url + '?t=' + Date.now(); })",
                                capture.snapshot_url,
                            )
                            jpeg = capture.page.screenshot(
                                type="jpeg", quality=cfg.dashboard.jpeg_quality,
                                full_page=False, timeout=5000,
                            )
                            frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is None:
                                logger.warning("[%s] Chromium entrego un screenshot invalido", capture.cam.tag)
                                continue
                            capture.dashboard_recorder.write(frame)
                            if cfg.type_enabled("iso"):
                                iso = crop_css_box(
                                    frame, capture.iso_box,
                                    cfg.dashboard.width, cfg.dashboard.height,
                                )
                                capture.iso_recorder.write(iso)
                        except Exception:
                            logger.exception("[%s] fallo capturando el panel web", capture.cam.tag)
                    stop.wait(max(0.0, period - (time.monotonic() - started)))
            finally:
                browser.close()
    except Exception:
        logger.exception("no se pudo iniciar Chromium para grabar ISO/dashboard")
    finally:
        for capture in captures:
            capture.iso_recorder.close()
            capture.dashboard_recorder.close()
