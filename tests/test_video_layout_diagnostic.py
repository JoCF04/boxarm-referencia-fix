from __future__ import annotations

"""Diagnostico reproducible de plantillas sobre frames aleatorios de video.

Uso manual (ejecuta YOLO y escribe JPG + summary.json)::

    python tests/test_video_layout_diagnostic.py \
        --video videos/1.00/p4r1.mp4 --samples 6 --seed 100 --device cpu

La prueba pesada de pytest es opt-in porque necesita pesos, video y varios
segundos de inferencia. La aleatoriedad siempre lleva semilla para poder
reproducir exactamente un frame que falle.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
from collections import Counter

import cv2
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(PROJECT_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def sample_frame_indices(
    total_frames: int,
    samples: int,
    seed: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[int, ...]:
    """Muestra indices unicos y ordenados dentro del intervalo solicitado."""
    stop = total_frames if end_frame is None else min(end_frame, total_frames)
    start = max(0, start_frame)
    if stop <= start:
        raise ValueError("el intervalo de video esta vacio")
    population = range(start, stop)
    count = min(max(1, samples), len(population))
    return tuple(sorted(random.Random(seed).sample(population, count)))


def test_random_frame_indices_are_reproducible_and_unique() -> None:
    first = sample_frame_indices(1000, samples=8, seed=100, start_frame=100)
    second = sample_frame_indices(1000, samples=8, seed=100, start_frame=100)

    assert first == second
    assert len(first) == len(set(first)) == 8
    assert all(100 <= index < 1000 for index in first)


def _names_by_id(names) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def _draw_report(
    frame: np.ndarray,
    roi: np.ndarray,
    detections: list[tuple[int, int, int, int, float, str]],
    report: dict,
) -> np.ndarray:
    canvas = frame.copy()
    cv2.polylines(canvas, [roi], True, (255, 200, 0), 2)
    for x1, y1, x2, y2, confidence, class_name in detections:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 210, 255), 2)
        cv2.putText(
            canvas,
            f"{class_name} {confidence:.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 210, 255),
            2,
        )

    panel_width = 410
    output = np.zeros((canvas.shape[0], canvas.shape[1] + panel_width, 3), dtype=np.uint8)
    output[:, : canvas.shape[1]] = canvas
    lines = [
        f"frame={report['frame_index']}  t={report['time_s']:.2f}s",
        f"estado={report['status']}",
        f"visibles={report['visible_detections']}",
        f"fase N0={report['phase']}",
        f"total reconstruido={report['total']}",
    ]
    for level, cells in report["levels"].items():
        lines.append(f"N{level} ({len(cells)}): {','.join(map(str, cells))}")
    if report["candidates"]:
        best = report["candidates"][0]
        lines.append(
            f"mejor candidato: N0={best['phase']} {best['hypothesis']} "
            f"e={best['mean_error']:.4f}"
        )
        for level, cells in best["levels"].items():
            lines.append(f"  cand N{level}: {','.join(map(str, cells))}")
    elif not report["arm_visible"]:
        lines.append("sin ajuste global A/B valido")
    if report["arm_visible"]:
        lines.append("BRAZO: frame descartado")
    y = 38
    for line in lines:
        cv2.putText(
            output,
            line,
            (canvas.shape[1] + 15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (235, 240, 245),
            1,
        )
        y += 30
    return output


def _candidate_layouts(counter, parsed: list[tuple]) -> list[dict]:
    """Enumera candidatos aunque el runtime los mantenga como ambiguos."""
    from boxarm.vision.palletizing.templates.template_matcher import (
        TemplateObservation,
        fit_layout_hypothesis,
    )

    consensus = counter._level_footprint.get(0)
    if consensus is None:
        return []
    box_class = counter._box_class or next(
        (class_name for _bbox, _confidence, class_name in parsed if class_name),
        "",
    )
    if not box_class:
        return []
    complete: list[TemplateObservation] = []
    partials: list[TemplateObservation] = []
    expected_long, expected_short = max(consensus), min(consensus)
    side_ratio = counter._cfg.min_complete_side_ratio
    for (x1, y1, x2, y2), _confidence, _class_name in parsed:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        measured = counter._measure_footprint(cx, cy, x2 - x1, y2 - y1)
        projected = cv2.perspectiveTransform(
            np.array([[[cx, cy]]], dtype=np.float32),
            counter._homography,
        )[0, 0]
        observation = TemplateObservation(
            cx,
            cy,
            x2 - x1,
            y2 - y1,
            canonical_u=float(projected[0]),
            canonical_v=float(projected[1]),
            canonical_width=measured[0],
            canonical_height=measured[1],
        )
        observed_long, observed_short = max(measured), min(measured)
        if (
            observed_long < expected_long * side_ratio
            or observed_short < expected_short * side_ratio
        ):
            partials.append(observation)
        else:
            complete.append(observation)

    candidates: list[dict] = []
    observations = tuple(complete)
    for phase in (0, 1):
        for include_upper in (False, True):
            fit = fit_layout_hypothesis(
                box_class,
                base_level=0,
                observations=observations,
                include_upper=include_upper,
                max_center_distance=counter._cfg.tau_cell,
                min_side_ratio=counter._cfg.min_complete_side_ratio,
                phase=phase,
            )
            if fit is None:
                continue
            if partials and (
                not include_upper
                or not counter._template_partials_are_explained(
                    fit,
                    observations,
                    tuple(partials),
                    base_level=0,
                )
            ):
                continue
            levels: dict[str, list[int]] = {}
            for assignment in fit.assignments:
                levels.setdefault(str(assignment.level), []).append(
                    assignment.slot.cell
                )
            for cells in levels.values():
                cells.sort()
            candidates.append({
                "phase": "A" if phase == 0 else "B",
                "hypothesis": "two_levels" if include_upper else "one_level",
                "mean_error": fit.mean_error,
                "complete": len(complete),
                "partials": len(partials),
                "levels": levels,
            })
    return sorted(
        candidates,
        key=lambda item: (
            item["mean_error"],
            1 if item["hypothesis"] == "two_levels" else 0,
            item["phase"],
        ),
    )


def analyze_random_video_frames(
    video: Path,
    output_dir: Path,
    *,
    samples: int = 6,
    seed: int = 100,
    device: str = "cpu",
    camera_id: int = 1,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> dict:
    """Ejecuta los detectores reales y resume la composicion de cada frame."""
    from boxarm.config import (
        load_palletizing_config,
        load_pipeline_config,
        load_vision_config,
    )
    from boxarm.vision.inference import (
        _arm_detections,
        _box_detections,
        _scale_roi,
        load_models,
    )
    from boxarm.vision.palletizing import GridCounter

    pipeline_cfg = load_pipeline_config(PROJECT_ROOT / "configs/pipeline.yaml")
    vision_cfg = load_vision_config(PROJECT_ROOT / "configs/vision.yaml")
    pallet_cfg = load_palletizing_config(PROJECT_ROOT / "configs/palletizing.yaml")
    try:
        camera = next(item for item in pipeline_cfg.cameras if item.id == camera_id)
    except StopIteration as exc:
        raise ValueError(f"camera_id inexistente: {camera_id}") from exc

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"no se pudo abrir el video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError("el video no informa FPS/frame_count validos")
    indices = sample_frame_indices(
        total_frames,
        samples,
        seed,
        start_frame=int(start_s * fps),
        end_frame=None if end_s is None else int(end_s * fps),
    )

    arm_model, box_model = load_models(vision_cfg)
    arm_names = _names_by_id(arm_model.names)
    box_names = _names_by_id(box_model.names)
    arm_class_id = next(
        class_id
        for class_id, name in arm_names.items()
        if name == vision_cfg.arm.class_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict] = []
    try:
        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"no se pudo leer el frame {frame_index}")
            frame_h, frame_w = frame.shape[:2]
            roi = _scale_roi(camera.roi, frame_w, frame_h)

            arm_result = arm_model.predict(
                frame,
                imgsz=vision_cfg.arm.imgsz,
                conf=vision_cfg.arm.conf,
                device=device,
                verbose=False,
            )[0]
            arm_boxes = _arm_detections(
                arm_result.boxes,
                arm_class_id=arm_class_id,
                conf_floor=vision_cfg.arm.conf,
            )
            detections: list[tuple[int, int, int, int, float, str]] = []
            counter = GridCounter(roi, pallet_cfg)
            if not arm_boxes:
                box_result = box_model.predict(
                    frame,
                    imgsz=vision_cfg.boxes.imgsz,
                    conf=vision_cfg.boxes.conf,
                    device=device,
                    verbose=False,
                )[0]
                detections = _box_detections(
                        box_result.boxes,
                        roi_pts=roi,
                        class_names=box_names,
                        class_conf=vision_cfg.boxes.class_conf,
                        roi_tolerance_ratio=vision_cfg.boxes.roi_tolerance_ratio,
                    )
                if detections:
                    active_class = Counter(item[5] for item in detections).most_common(1)[0][0]
                    counter.set_box_class(active_class)
                    detections = [item for item in detections if item[5] == active_class]
                    footprints = []
                    for x1, y1, x2, y2, _confidence, _class_name in detections:
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        footprints.append(counter._measure_footprint(cx, cy, x2 - x1, y2 - y1))
                    counter._level_footprint[0] = (
                        float(np.median([item[0] for item in footprints])),
                        float(np.median([item[1] for item in footprints])),
                    )
                parsed = [
                    ((x1, y1, x2, y2), confidence, class_name)
                    for x1, y1, x2, y2, confidence, class_name in detections
                ]
                candidates = _candidate_layouts(counter, parsed)
                # Diagnostico geometrico de un frame congelado: se repite la
                # misma evidencia para satisfacer la barrera temporal. Esto
                # NO simula movimiento ni sustituye una corrida secuencial.
                for _ in range(pallet_cfg.confirmation.min_stable):
                    counter._reconcile_initial_layers(parsed)
            else:
                candidates = []

            levels: dict[str, list[int]] = {}
            for cell, level in sorted(counter._occupied, key=lambda item: (item[1], item[0])):
                levels.setdefault(str(level), []).append(cell)
            phase = (
                None
                if counter._template_phase is None
                else ("A" if counter._template_phase == 0 else "B")
            )
            status = (
                "arm_pause"
                if arm_boxes
                else ("resolved" if counter._bootstrap_reconciled else "waiting_or_ambiguous")
            )
            diagnostic = (
                "frame descartado por brazo"
                if arm_boxes
                else (
                    "ajuste global disponible"
                    if candidates
                    else "ninguna hipotesis A/B explica todas las detecciones; "
                    "revisar duplicados, ROI o centroides de plantilla"
                )
            )
            report = {
                "frame_index": frame_index,
                "time_s": frame_index / fps,
                "status": status,
                "arm_visible": bool(arm_boxes),
                "visible_detections": len(detections),
                "phase": phase,
                "total": counter.total,
                "levels": levels,
                "candidates": candidates,
                "diagnostic": diagnostic,
            }
            reports.append(report)
            rendered = _draw_report(frame, roi, detections, report)
            cv2.imwrite(
                str(output_dir / f"frame_{frame_index:07d}_{status}.jpg"),
                rendered,
            )
    finally:
        cap.release()

    summary = {
        "video": str(video.resolve()),
        "seed": seed,
        "samples": len(reports),
        "warning": (
            "Cada frame se analiza congelado; waiting_or_ambiguous es un resultado valido. "
            "La confirmacion real requiere frames consecutivos de la escena."
        ),
        "frames": reports,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


@pytest.mark.skipif(
    os.environ.get("RUN_VIDEO_LAYOUT_DIAGNOSTIC") != "1",
    reason="diagnostico YOLO opt-in; definir RUN_VIDEO_LAYOUT_DIAGNOSTIC=1",
)
def test_random_real_video_frames_report_layout(tmp_path: Path) -> None:
    video = Path(
        os.environ.get(
            "PALLET_VIDEO",
            PROJECT_ROOT / "videos/1.00/p4r1.mp4",
        )
    )
    summary = analyze_random_video_frames(
        video,
        tmp_path,
        samples=int(os.environ.get("PALLET_RANDOM_FRAMES", "3")),
        seed=int(os.environ.get("PALLET_RANDOM_SEED", "100")),
        device=os.environ.get("PALLET_DEVICE", "cpu"),
    )

    assert summary["samples"] == int(os.environ.get("PALLET_RANDOM_FRAMES", "3"))
    assert len({item["frame_index"] for item in summary["frames"]}) == summary["samples"]
    assert (tmp_path / "summary.json").exists()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Muestrea frames aleatorios y reporta fase/niveles/celdas del pallet.",
    )
    parser.add_argument("--video", type=Path, default=PROJECT_ROOT / "videos/1.00/p4r1.mp4")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/video_layout_diagnostic")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera-id", type=int, default=1)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float)
    args = parser.parse_args()
    summary = analyze_random_video_frames(
        args.video,
        args.output_dir,
        samples=args.samples,
        seed=args.seed,
        device=args.device,
        camera_id=args.camera_id,
        start_s=args.start_s,
        end_s=args.end_s,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artefactos: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
