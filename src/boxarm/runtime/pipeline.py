from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : Orquestacion: un proceso por camara (multiprocessing), un
#              hilo drain por camara y un unico Flask -- reemplaza la
#              version anterior de hilos + N apps Flask.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : run() recibe VisionConfig y DrawingConfig por separado
#              (configs/vision.yaml, configs/drawing.yaml) y usa
#              cfg.runtime para los timeouts de apagado/loop principal
#              en vez de constantes sueltas (G-5).
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : El ISO web usa solo geometria JSON y render local
#              para la vista de inspeccion 3D en /cam/<id>/iso, en
#              paralelo al stream normal.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : el loop principal supervisa los procesos-camara y relanza el
#              que muere (hasta runtime.max_camera_restarts). Antes un
#              proceso caido dejaba su stream congelado para siempre y el
#              resto del sistema seguia como si nada.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : el aviso de proceso caido incluye el exitcode. Un traceback
#              del hijo no cruza al padre, asi que sin esto una muerte por
#              OOM (senal 9) o segfault (senal 11) era indistinguible de
#              una excepcion de Python.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 25-08-26
# Reason     : el supervisor relanzaba tambien los procesos que salian con
#              exitcode 0, asi que con loop_video=false el video igual
#              volvia a empezar sin fin. Salida limpia = fuente agotada.
# -----------------------------------------------------------------------
"""run(cfg, vision_cfg, drawing_cfg, palletizing_cfg) arranca todo: un
multiprocessing.Process por camara, dos hilos drain-<tag> por camara
(normal + iso) en el proceso principal, y el unico Flask que sirve las N
camaras. Apagado limpio con Ctrl-C (P-16)."""

import logging
import multiprocessing as mp
import threading
import time

import cv2
import torch

from boxarm.config import DrawingConfig, IsometricConfig, PalletizingConfig, PipelineConfig, VisionConfig
from boxarm.runtime.web_recording import record_web_views
from boxarm.web.streaming import (drain_jpeg_queue, make_flask_app,
                                   drain_iso_scene_queue)
from boxarm.runtime.workers import camera_worker

logger = logging.getLogger(__name__)


def _spawn_camera(ctx, cam, cfg, vision_cfg, drawing_cfg, palletizing_cfg,
                  isometric_cfg, jpeg_q, iso_scene_q, stop) -> mp.Process:
    """Arranca el proceso de UNA camara sobre colas ya creadas. Se llama al
    inicio y otra vez desde el supervisor cuando hay que relanzar: las colas
    y la cola de geometria sobreviven al proceso, asi que Flask y el renderer
    del navegador siguen apuntando al mismo estado."""
    proc = ctx.Process(
        target=camera_worker,
        args=(cam, cfg, vision_cfg, drawing_cfg, palletizing_cfg, isometric_cfg,
              jpeg_q, iso_scene_q, stop),
        daemon=True, name=f"camera-{cam.tag}",
    )
    proc.start()
    return proc


def run(cfg: PipelineConfig,
        vision_cfg: VisionConfig,
        drawing_cfg: DrawingConfig,
        palletizing_cfg: PalletizingConfig,
        isometric_cfg: IsometricConfig) -> None:
    n = cv2.getNumberOfCPUs()
    cv2.setNumThreads(n)
    torch.set_num_threads(n)

    logger.info("[MODO] %s", cfg.modo)

    enabled_cameras = tuple(cam for cam in cfg.cameras if cam.enabled)
    # 0.0.0.0/:: sirven para escuchar en todas las interfaces, pero no son
    # una direccion util para abrir el panel desde el navegador local.
    browser_host = "127.0.0.1" if cfg.web.flask_host in {"0.0.0.0", "::"} else cfg.web.flask_host
    for cam in cfg.cameras:
        if not cam.enabled:
            logger.info("[%s] enabled: false -- no arranca (configs/pipeline.yaml)", cam.tag)

    # spawn, no el fork() por default en Linux: el padre ya inicializo
    # torch/cv2 (CUDA, hilos) arriba -- forkear un hijo con eso a medias
    # rompe NVDEC/V4L2 en el hijo (pipeline GStreamer que nunca llega a
    # PLAYING, aunque el mismo pipeline funcione suelto en un proceso
    # nuevo). spawn arranca cada proceso-camara limpio.
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    # Todo lo que el supervisor necesita para relanzar una camara: sus colas,
    # su cola de geometria y cuantas veces ya se relanzo.
    supervised: list[dict] = []
    drain_threads: list[threading.Thread] = []

    for cam in enabled_cameras:
        jpeg_q     = ctx.Queue(maxsize=2)
        iso_scene_q = ctx.Queue(maxsize=2)

        proc = _spawn_camera(ctx, cam, cfg, vision_cfg, drawing_cfg, palletizing_cfg,
                             isometric_cfg, jpeg_q, iso_scene_q, stop)
        supervised.append({
            "cam": cam, "proc": proc, "jpeg_q": jpeg_q,
            "iso_scene_q": iso_scene_q,
            "restarts": 0, "dead_since": None,
        })

        t_drain = threading.Thread(
            target=drain_jpeg_queue, args=(cam.id, jpeg_q, cfg.runtime, stop),
            daemon=True, name=f"drain-{cam.tag}",
        )
        t_drain.start()
        drain_threads.append(t_drain)

        t_drain_scene = threading.Thread(
            target=drain_iso_scene_queue, args=(cam.id, iso_scene_q, cfg.runtime, stop),
            daemon=True, name=f"drain-scene-{cam.tag}",
        )
        t_drain_scene.start()
        drain_threads.append(t_drain_scene)

        logger.info(
            "[%s] proceso arrancado -- panel: http://%s:%d/cam/%d (bind=%s)",
            cam.tag, browser_host, cfg.web.port, cam.id, cfg.web.flask_host,
        )

    # La web conoce también las cámaras deshabilitadas para poder mostrarlas
    # como estado operativo; solo las habilitadas tienen proceso y drains.
    app = make_flask_app(cfg.cameras, cfg.runtime,
                         (isometric_cfg.azimuth_deg, isometric_cfg.elevation_deg),
                         isometric_cfg, drawing_cfg,
                         stream_max_fps=cfg.web.stream_max_fps)
    t_flask = threading.Thread(
        target=lambda: app.run(host=cfg.web.flask_host, port=cfg.web.port, threaded=True, use_reloader=False),
        daemon=True, name="flask",
    )
    t_flask.start()
    logger.info(
        "Flask bind=%s:%d -- abra http://%s:%d/ -- Ctrl+C para salir.",
        cfg.web.flask_host, cfg.web.port, browser_host, cfg.web.port,
    )

    recording_threads: list[threading.Thread] = []
    if cfg.recording.type_enabled("iso") or cfg.recording.type_enabled("dashboard"):
        t_web_recording = threading.Thread(
            target=record_web_views,
            args=(enabled_cameras, cfg.recording, cfg.web.port, stop),
            daemon=True,
            name="record-web-views",
        )
        t_web_recording.start()
        recording_threads.append(t_web_recording)

    # Supervisor: una camara caida se relanza sola, las demas ni se enteran.
    # El limite existe para no entrar en un lazo de relanzamientos cuando el
    # fallo es de configuracion (pesos ilegibles, /dev/video inexistente) y
    # volveria a morir igual.
    try:
        while not stop.is_set():
            stop.wait(cfg.runtime.main_loop_tick_s)
            if stop.is_set():
                break
            now = time.monotonic()
            for entry in supervised:
                if entry["proc"].is_alive():
                    entry["dead_since"] = None
                    continue
                tag = entry["cam"].tag
                # exitcode negativo = el proceso no termino solo, lo mato una
                # senal: -9 es el OOM killer, -11 un segfault (tipicamente en
                # GStreamer/NVDEC). Un traceback de Python nunca llega aqui, se
                # ve en el log del hijo, asi que sin esto la muerte es opaca.
                code = entry["proc"].exitcode
                # exitcode 0 = el proceso termino solo y sin error: en modo
                # video con loop_video=false eso es simplemente "se acabo el
                # archivo". Relanzarlo hacia que el video volviera a empezar
                # una y otra vez, o sea justo lo que loop_video=false pide
                # que NO pase.
                if code == 0:
                    if entry["dead_since"] is None:
                        entry["dead_since"] = now
                        logger.info("[%s] proceso terminado sin error -- fuente agotada, no se relanza", tag)
                    continue
                if entry["dead_since"] is None:
                    motivo = f"senal {-code}" if code is not None and code < 0 else f"exitcode {code}"
                    logger.warning("[%s] proceso caido (%s)", tag, motivo)
                if entry["restarts"] >= cfg.runtime.max_camera_restarts:
                    if entry["dead_since"] is not None:
                        continue  # ya se aviso, no repetir el log en cada tick
                    entry["dead_since"] = now
                    logger.error("[%s] proceso caido y sin relanzamientos restantes (%d) "
                                 "-- esta camara queda fuera", tag, cfg.runtime.max_camera_restarts)
                    continue
                # Espera antes de relanzar: si la causa es una camara USB que
                # se desconecto, reintentar al instante solo gasta arranques.
                if entry["dead_since"] is None:
                    entry["dead_since"] = now
                    logger.warning("[%s] relanzando en %.1fs",
                                   tag, cfg.runtime.camera_restart_delay_s)
                    continue
                if now - entry["dead_since"] < cfg.runtime.camera_restart_delay_s:
                    continue
                entry["proc"].join(timeout=0)
                entry["restarts"] += 1
                entry["proc"] = _spawn_camera(
                    ctx, entry["cam"], cfg, vision_cfg, drawing_cfg, palletizing_cfg,
                    isometric_cfg, entry["jpeg_q"], entry["iso_scene_q"], stop)
                entry["dead_since"] = None
                logger.info("[%s] proceso relanzado (%d/%d)",
                            tag, entry["restarts"], cfg.runtime.max_camera_restarts)
    except KeyboardInterrupt:
        logger.info("Deteniendo...")
    finally:
        stop.set()
        web_outputs = sum((
            cfg.recording.type_enabled("iso"),
            cfg.recording.type_enabled("dashboard"),
        ))
        web_recording_timeout = cfg.runtime.shutdown_join_timeout_s
        if cfg.recording.transcode_h264:
            web_recording_timeout += web_outputs * cfg.recording.transcode_timeout_s
        for t in recording_threads:
            logger.info("esperando cierre y conversion H.264 de las grabaciones web...")
            t.join(timeout=web_recording_timeout)
            if t.is_alive():
                logger.error("[%s] no termino en %.1fs; revise FFmpeg y los .tmp.mp4",
                             t.name, web_recording_timeout)
        for t in drain_threads:
            t.join(timeout=cfg.runtime.shutdown_join_timeout_s)
        for entry in supervised:
            proc = entry["proc"]
            proc.join(timeout=cfg.runtime.shutdown_join_timeout_s)
            if proc.is_alive():
                logger.warning("[%s] no termino a tiempo, forzando cierre", proc.name)
                proc.terminate()
