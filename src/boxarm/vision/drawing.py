from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm -- Changelog
# Programmer  | Date     | Resumen
# ----------- | -------- | -----------------------------------------------
# gerald      | 23-08-26 | Dibujado (ROI, cajas, HUD) extraido del bucle de
#             |          | tracking a su propio modulo: separa "que se
#             |          | detecta" de "que se pinta", sin cambios de
#             |          | comportamiento.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : Colores y layout (grosores, escalas de fuente, tamano del
#              HUD) ya no hardcodeados -- vienen de DrawingConfig, cargado
#              desde configs/drawing.yaml (G-5).
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : CellState/GridDetection ahora se importan de
#              boxarm.vision.palletizing (grid_counting.py renombrado al
#              unificar el cerebro del conteo). Sin cambios de dibujado.
# -----------------------------------------------------------------------
"""Anotaciones sobre el frame: ROI, cajas detectadas y HUD. No decide
nada de conteo ni de deteccion, solo pinta sobre un frame ya procesado.
Colores y layout vienen de DrawingConfig (configs/drawing.yaml)."""

from typing import Protocol

import cv2
import numpy as np

from boxarm.config import DrawingConfig
from boxarm.vision.palletizing import CellState, GridDetection


class _Counter(Protocol):
    """Interfaz minima que el HUD necesita del contador."""
    total: int
    visible: int


def color_for_level(level: int, cfg: DrawingConfig, brightness: float = 1.0) -> tuple[int, int, int]:
    """Color BGR estable por nivel, compartido por el overlay 2D y el ISO.

    ``brightness`` permite crear variantes del MISMO tono sin confundir
    nivel con estado: 1.0 para una caja nueva y una variante mas oscura
    para una re-deteccion.
    """
    if not cfg.level_colors:
        return cfg.color_new
    base = cfg.level_colors[level % len(cfg.level_colors)]
    factor = max(0.0, min(brightness, 1.0))
    return tuple(int(channel * factor) for channel in base)


def draw_roi(frame: np.ndarray, roi_pts: np.ndarray, cfg: DrawingConfig) -> None:
    cv2.polylines(frame, [roi_pts], isClosed=True, color=cfg.color_roi, thickness=cfg.roi_thickness)


def draw_grid_detections(frame: np.ndarray, results: list[GridDetection], cfg: DrawingConfig) -> None:
    """Dibuja cada deteccion segun nivel y resultado de GridCounter.update().

    Las aceptadas conservan un color base por nivel; NEW usa el tono
    completo y REDET una variante mas oscura -- el estado ya se lee en el
    color, no hace falta escribirlo. El texto queda en "celda N nivel"
    (mismo formato que el ISO) y solo la caja NUEVA se marca aparte.
    Las rechazadas van en rojo con el codigo del motivo, no la razon
    completa: con varias rechazadas superpuestas un texto largo por caja
    se vuelve ilegible."""
    for res in results:
        # Una observacion parcial usada para probar la relacion i -> i+1 no
        # es una caja dibujable. Pintarla produciria precisamente el bloque
        # encogido que la geometria persistente evita crear.
        if res.state is CellState.VALIDATION:
            continue
        x1, y1, x2, y2 = res.bbox
        if res.state is CellState.REJECTED:
            col = cfg.color_pending
            conf_txt = f" {res.confidence:.2f}" if res.confidence is not None else ""
            label      = f"{res.reason or '?'}{conf_txt}"
            text_color = col
            font_scale = cfg.box_label_font_scale * 0.7
        else:
            assert res.level is not None
            brightness = 1.0 if res.state is CellState.NEW else 0.68
            col = color_for_level(res.level, cfg, brightness)
            # Solo celda y nivel, con el mismo formato que el ISO ("7N1"),
            # para poder cruzar las dos vistas de un vistazo. Antes se
            # escribia ademas el estado, la confianza y el id de tracker:
            # con 16 cajas eso es una pared de texto sobre la paleta, y
            # "redet" es el caso normal -- casi todas las cajas del frame
            # son re-detecciones, asi que decirlo no informa nada. Solo se
            # marca la NUEVA, que es lo que cambia el conteo; confianza y
            # track siguen en el log, que es donde se los va a buscar.
            label = f"{res.cell}N{res.level}"
            if res.state is CellState.NEW:
                label += " NUEVA"
            text_color = col
            font_scale = cfg.box_label_font_scale
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, cfg.box_thickness)
        # Texto NEGRO con halo del color del nivel. El texto pintado en el
        # color del nivel se perdia contra las cajas claras -- sobre el
        # carton amarillo, un label amarillo es invisible. El halo mantiene
        # la codificacion por nivel sin sacrificar la legibilidad.
        pos = (x1, max(y1 - 6, 0))
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    text_color, cfg.box_label_thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), cfg.box_label_thickness, cv2.LINE_AA)
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), cfg.box_circle_radius, col, -1)


def draw_arm_present(
    frame: np.ndarray,
    roi_pts: np.ndarray,
    cfg: DrawingConfig,
    arm_bboxes: list[tuple[int, int, int, int]] | None = None,
) -> None:
    """Brazo detectado en el frame: dibuja solo el robot, nunca las cajas.

    El bbox hace visible POR QUE el gate esta cerrado. Antes solo aparecia
    la palabra BRAZO y no se podia verificar si la deteccion realmente
    fue la clase que cerro el gate.
    """
    for x1, y1, x2, y2 in arm_bboxes or []:
        cv2.rectangle(frame, (x1, y1), (x2, y2), cfg.color_arm_alert, cfg.box_thickness)
        cv2.putText(frame, "ROBOT", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.box_label_font_scale,
                    cfg.color_arm_alert, cfg.box_label_thickness, cv2.LINE_AA)
    cv2.putText(frame, "BRAZO",
                (roi_pts[0][0], roi_pts[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, cfg.arm_alert_font_scale,
                cfg.color_arm_alert, cfg.arm_alert_thickness)


def draw_hud(frame: np.ndarray, tag: str, counter: _Counter, inf_fps: float, cfg: DrawingConfig,
             *, box_class: str = "", level_counts: dict[int, int] | None = None,
             cam_fps: float | None = None, validating_initial: bool = False,
             observed_count: int = 0) -> None:
    """`box_class`/`level_counts` vienen del SceneState del frame (una sola
    denominacion activa a la vez, ver counter.set_box_class): sin esto el
    HUD solo decia CUANTAS cajas hay, no QUE se esta paletizando ni como
    se reparten por nivel -- lo mismo que ya muestra el ISO, resumido aca
    para no tener que abrir la vista 3D para leerlo.

    `cam_fps` es el fps de LECTURA de camara/video, separado de `inf_fps`
    (el del lazo de inferencia): con subsample_factor > 1 o un modelo
    lento, el segundo cae por debajo del primero y sin las dos lineas no
    se distingue "la camara entrega poco" de "el modelo no da abasto"."""
    cv2.rectangle(frame, (0, 0), (cfg.hud_width, cfg.hud_height), cfg.hud_background, -1)
    cv2.putText(frame, tag,
                (8,  24), cv2.FONT_HERSHEY_DUPLEX, cfg.hud_title_font_scale,
                cfg.color_hud_title, cfg.hud_title_thickness)
    # "En paleta" es el total que hay; "Colocadas" solo lo que el brazo puso
    # mientras mirabamos. Antes una sola linea decia "Colocadas: total" y
    # mezclaba las dos cosas: en un video que arranca con la paleta a medio
    # cargar, la mayor parte de ese numero nunca la coloco el brazo.
    inicial = getattr(counter, "initial", None)
    colocadas = getattr(counter, "placed", None)
    primary_text = (
        f"Analizando: {observed_count}"
        if validating_initial
        else f"En paleta : {counter.total}"
    )
    cv2.putText(frame, primary_text,
                (8,  56), cv2.FONT_HERSHEY_DUPLEX, cfg.hud_line_font_scale,
                cfg.color_new, cfg.hud_line_thickness)
    if validating_initial:
        cv2.putText(frame, "  estado inicial sin confirmar",
                    (8,  82), cv2.FONT_HERSHEY_DUPLEX, cfg.hud_line_font_scale * 0.8,
                    cfg.color_text, cfg.hud_visible_thickness)
    elif inicial is not None and colocadas is not None:
        cv2.putText(frame, f"  inicial {inicial} + brazo {colocadas}",
                    (8,  82), cv2.FONT_HERSHEY_DUPLEX, cfg.hud_line_font_scale * 0.8,
                    cfg.color_text, cfg.hud_visible_thickness)
    cv2.putText(frame, f"Visibles  : {counter.visible}",
                (8,  108), cv2.FONT_HERSHEY_DUPLEX, cfg.hud_line_font_scale,
                cfg.color_text, cfg.hud_visible_thickness)

    y = 136
    if box_class:
        cv2.putText(frame, f"Clase     : {box_class}",
                    (8, y), cv2.FONT_HERSHEY_DUPLEX, cfg.hud_line_font_scale * 0.8,
                    cfg.color_text, cfg.hud_visible_thickness)
        y += 26
    if level_counts:
        breakdown = "  ".join(f"N{level}:{count}" for level, count in sorted(level_counts.items()))
        cv2.putText(frame, breakdown,
                    (8, y), cv2.FONT_HERSHEY_DUPLEX, cfg.hud_line_font_scale * 0.8,
                    cfg.color_text, cfg.hud_visible_thickness)
        y += 26

    cv2.putText(frame, f"FPS camara: {cam_fps:.1f}" if cam_fps is not None else "FPS camara: --",
                (8, y), cv2.FONT_HERSHEY_DUPLEX, cfg.hud_fps_font_scale,
                cfg.color_text, cfg.hud_fps_thickness)
    y += 24
    cv2.putText(frame, f"FPS infer : {inf_fps:.1f}",
                (8, y), cv2.FONT_HERSHEY_DUPLEX, cfg.hud_fps_font_scale,
                cfg.color_text, cfg.hud_fps_thickness)
