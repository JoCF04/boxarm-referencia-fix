from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm -- Changelog
# Programmer  | Date     | Resumen
# ----------- | -------- | -----------------------------------------------
# gerald      | 23-08-26 | Bucle de inferencia YOLO frame-a-frame aislado de
#             |          | captura y dibujado, frames por queue.Queue.
# gerald      | 23-08-26 | Configuracion de deteccion y fps_smoothing_alpha desde
#             |          | PipelineConfig.runtime en vez de hardcodeados.
# gerald      | 23-08-26 | Despacha a GridCounter (identidad por celda y
#             |          | nivel, sin identidad temporal).
# gerald      | 23-08-26 | Vista de inspeccion 3D en un segundo jpeg_out,
#             |          | grabacion opcional de ambos streams y CameraConfig
#             |          | completa en vez de roi_pts/tag sueltos.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : recibe IsometricConfig por separado (configs/isometric.yaml)
#              -- geometria del render 3D ya no vive dentro de
#              PalletizingConfig (separacion de responsabilidades).
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : tracking.py -> inference.py y run_tracker -> run_inference.
#              El modulo queda como I/O puro: mide (motion_score) y observa
#              (arm_visible, bboxes recortadas al ROI) pero ya no decide --
#              el gate, el ciclo del brazo y el diagnostico de celdas se
#              mudaron a palletizing.GridCounter.update(). El ISO se
#              renderiza desde el SceneState que entrega el cerebro.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : log periodico de rendimiento (runtime.fps_log_interval_s):
#              fps real del lazo y reparto de ms por etapa (espera,
#              yolo, motion, dibujo, jpeg, iso). El HUD ya mostraba
#              un fps suavizado, pero no decia DONDE se va el tiempo.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : la carga del modelo sale del hilo a load_model(), que llama
#              camera_worker antes de arrancarlo. Dentro del hilo, un .pt
#              incompatible mataba solo al hilo y el proceso seguia
#              sirviendo el stream sin inferencia, en silencio.
# -----------------------------------------------------------------------
"""Inferencia YOLO frame a frame mediante ``model.predict()``.

Este modulo no conserva IDs ni toma decisiones de conteo: entrega la
observacion cruda a ``GridCounter`` y dibuja el estado que este resuelve.
"""

import logging
import queue
import time
from dataclasses import replace

import cv2
import numpy as np

from boxarm.capture.camera_io import push_frame
from boxarm.config import CameraConfig, DrawingConfig, IsometricConfig, PalletizingConfig, PipelineConfig, VisionConfig
from boxarm.runtime.recording import VideoRecorder
from boxarm.vision.palletizing import (
    FrameInput,
    GateState,
    GridCounter,
    SceneBox,
    SceneState,
)
from boxarm.vision import drawing

logger = logging.getLogger(__name__)


def _resolve_class_ids(cls_names, names: tuple[str, ...], *, where: str) -> tuple[int, ...]:
    """Resuelve NOMBRES configurados contra las clases de UN modelo.

    Se resuelve por nombre a proposito. El id de una clase depende del orden
    del `names:` del dataset y se corre al agregar etiquetas nuevas: una
    config por id seguia siendo "valida" (el id existia) pero apuntaba a otra
    clase. Un nombre ausente, en cambio, falla aqui de forma ruidosa al
    arrancar."""
    id_by_name = {str(name): int(class_id) for class_id, name in cls_names.items()}
    unknown = [name for name in names if name not in id_by_name]
    if unknown:
        raise ValueError(
            f"{where}: clases configuradas que no existen en el modelo: "
            f"{unknown}; el modelo expone {sorted(id_by_name)}. Si acabas de "
            f"reentrenar, revisa los nombres en configs/vision.yaml"
        )
    return tuple(id_by_name[name] for name in names)





def _arm_detections(boxes, *, arm_class_id: int, conf_floor: float) -> list[tuple[int, int, int, int]]:
    """bboxes del detector dedicado del brazo (1 sola clase) que superan
    conf_floor. Sin ROI: el brazo puede entrar/salir por cualquier borde del
    frame, no solo por donde esta la paleta."""
    # `int(box.cls[0])`, `float(box.conf[0])` and `map(int, box.xyxy[0])`
    # request individual values from the result tensor.  When Ultralytics
    # keeps that tensor on CUDA, every request can synchronize the CPU with
    # the GPU.  Filter the complete tensors first and cross the CPU boundary
    # once for the selected detections.  The public return contract remains
    # the same list of integer tuples.
    xyxy = boxes.xyxy
    cls = boxes.cls
    conf = boxes.conf
    if xyxy is None or cls is None or conf is None or len(xyxy) == 0:
        return []

    selected = (cls == arm_class_id) & (conf >= conf_floor)
    selected_xyxy = xyxy[selected].detach().cpu().numpy()
    return [tuple(int(value) for value in row) for row in selected_xyxy]


def _bbox_fully_inside_roi(
    bbox: tuple[int, int, int, int],
    roi_pts: np.ndarray,
    tolerance_px: float = 0.0,
) -> bool:
    """True cuando las cuatro esquinas estan dentro, con margen opcional."""
    x1, y1, x2, y2 = bbox
    return all(
        cv2.pointPolygonTest(roi_pts, (float(x), float(y)), True) >= -tolerance_px
        for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    )


def _box_detections(
    boxes,
    *,
    roi_pts: np.ndarray,
    class_names: dict[int, str],
    class_conf: dict[str, float] | None = None,
    roi_tolerance_ratio: float = 0.0,
) -> list[tuple[int, int, int, int, float, str]]:
    """bboxes del detector de cajas (multiclase) recortadas al ROI de la
    paleta, con umbral por clase. Solo se llama cuando el detector del
    brazo NO vio brazo en el frame (ver run_inference)."""
    thresholds = class_conf or {}
    detections: list[tuple[int, int, int, int, float, str]] = []
    xyxy = boxes.xyxy
    cls = boxes.cls
    conf = boxes.conf
    if xyxy is None or cls is None or conf is None or len(xyxy) == 0:
        return detections

    # Class/confidence filtering stays in the result tensor.  The configured
    # thresholds are per class, so build the mask by class (this loop is over
    # class definitions, never over detections).  The fallback preserves the
    # previous behavior for an unknown class: threshold 0.0.
    import torch

    if thresholds:
        selected = torch.zeros_like(conf, dtype=torch.bool)
        known_class = torch.zeros_like(cls, dtype=torch.bool)
        for class_id, class_name in class_names.items():
            class_mask = cls == class_id
            known_class |= class_mask
            selected |= class_mask & (conf >= thresholds.get(class_name, 0.0))
        selected |= (~known_class) & (conf >= 0.0)
    else:
        selected = conf >= 0.0

    # One grouped D2H copy for all fields needed by the existing Python/OpenCV
    # contract.  Avoid checking `selected.any()` on CUDA: that check itself
    # would synchronize before the grouped transfer.
    selected_rows = torch.cat(
        (xyxy[selected], conf[selected].unsqueeze(1), cls[selected].unsqueeze(1)), dim=1,
    ).detach().cpu().numpy()
    candidates = []
    for x1_f, y1_f, x2_f, y2_f, conf_f, cls_f in selected_rows:
        cls_id = int(cls_f)
        class_name = class_names.get(cls_id, str(cls_id))
        conf = float(conf_f)
        if conf < thresholds.get(class_name, 0.0):
            continue
        x1, y1, x2, y2 = int(x1_f), int(y1_f), int(x2_f), int(y2_f)
        candidates.append((x1, y1, x2, y2, conf, class_name))

    median_side = float(np.median([
        max(x2 - x1, y2 - y1) for x1, y1, x2, y2, _conf, _name in candidates
    ])) if candidates else 0.0
    tolerance_px = max(0.0, float(roi_tolerance_ratio)) * median_side
    for x1, y1, x2, y2, conf, class_name in candidates:
        if not _bbox_fully_inside_roi((x1, y1, x2, y2), roi_pts, tolerance_px):
            continue
        detections.append((x1, y1, x2, y2, conf, class_name))
    return detections


def _roi_motion_score(prev_gray: np.ndarray | None,
                      gray: np.ndarray,
                      roi_mask: np.ndarray,
                      diff: np.ndarray | None = None) -> float:
    """Movimiento promedio dentro del ROI usando diferencia absoluta entre
    frames consecutivos. No intenta entender objetos ni umbraliza nada: es
    una medicion cruda de pixeles que se entrega tal cual al contador."""
    if prev_gray is None:
        return 0.0
    if diff is None:
        diff = cv2.absdiff(prev_gray, gray)
    else:
        cv2.absdiff(prev_gray, gray, dst=diff)
    return float(cv2.mean(diff, mask=roi_mask)[0])


def _deliver_latest(output, data: object) -> bool:
    """Publica el ultimo SceneState sin bloquear la inferencia."""
    try:
        output.put_nowait(data)
        return True
    except queue.Full:
        pass
    for _ in range(2):
        try:
            output.get_nowait()
        except (queue.Empty, OSError, EOFError):
            break
        try:
            output.put_nowait(data)
            return True
        except queue.Full:
            continue
    return False


def _confirmed_scene_signature(scene: SceneState) -> tuple:
    """Firma estable de la verdad confirmada y nada mas.

    Persistencia y renderer ISO legado dependen de esta firma. En particular,
    una observacion ``confirming`` no puede provocar escrituras de estado ni
    un raster 3D costoso mientras todavia cambia frame a frame.
    """
    # Durante bootstrap las cotas se calculan a partir de identidades
    # internas que todavia pueden moverse entre niveles. No pertenecen a la
    # firma confirmada aunque SceneState las conserve para el visor web.
    level_tops = () if scene.validating_initial else tuple(scene.level_tops)
    total_height = 0.0 if scene.validating_initial else scene.total_height
    levels = 0 if scene.validating_initial else scene.levels
    return (
        scene.total,
        scene.initial,
        scene.placed,
        len(scene.boxes),
        tuple(
            (
                box.cell,
                box.level,
                box.u,
                box.v,
                box.side_a,
                box.side_b,
                box.z0,
                box.height,
                box.box_class,
            )
            for box in scene.boxes
        ),
        tuple(
            (
                overlap.cell_a,
                overlap.cell_b,
                overlap.level,
                overlap.ratio,
                overlap.u0,
                overlap.v0,
                overlap.u1,
                overlap.v1,
                overlap.z0,
                overlap.height,
            )
            for overlap in scene.overlaps
        ),
        level_tops,
        total_height,
        levels,
    )


def _web_scene_signature(scene: SceneState) -> tuple:
    """Firma de visualizacion: confirmadas + observaciones provisionales."""
    provisional = tuple(
        (
            box.cell,
            box.level,
            box.u,
            box.v,
            box.side_a,
            box.side_b,
            box.z0,
            box.height,
            box.box_class,
            box.status,
        )
        for box in scene.provisional_boxes
    )
    return _confirmed_scene_signature(scene) + (
        provisional,
        scene.validating_initial,
        tuple(scene.level_tops),
        scene.total_height,
        scene.levels,
    )


def _with_provisional_boxes(
    scene: SceneState,
    observed: list[SceneBox],
) -> SceneState:
    """Adjunta geometria visual y extiende las cotas sin volverla conteo."""
    if scene.validating_initial:
        # El bootstrap y el tracking son estados mutuamente excluyentes. En
        # arranque, la observacion cruda completa es la fuente visual de ESTA
        # revision; la hipotesis interna solo sirve como fallback cuando el
        # frame no contiene detecciones. Mezclarlas duplicaba objetos fisicos
        # como ANALIZANDO + CONFIRMANDO.
        provisional = (
            [replace(box, status="initializing") for box in observed]
            if observed
            else list(scene.provisional_boxes)
        )
    else:
        if not observed:
            return scene
        provisional = [*scene.provisional_boxes, *observed]

    if not provisional:
        return replace(scene, provisional_boxes=[])
    levels = max(scene.levels, max(box.level for box in provisional) + 1)
    level_tops = list(scene.level_tops)
    if not level_tops:
        level_tops = [0.0]
    while len(level_tops) < levels + 1:
        level = len(level_tops) - 1
        level_boxes = [box for box in provisional if box.level == level]
        fallback_height = level_boxes[0].height if level_boxes else 0.0
        top = max(
            (box.z0 + box.height for box in level_boxes),
            default=level_tops[-1] + fallback_height,
        )
        level_tops.append(max(level_tops[-1], top))

    total_height = max(
        scene.total_height,
        max((box.z0 + box.height for box in provisional), default=0.0),
        level_tops[-1],
    )
    return replace(
        scene,
        provisional_boxes=provisional,
        level_tops=level_tops,
        total_height=total_height,
        levels=levels,
    )


def _scale_roi(roi_norm: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    """CameraConfig.roi vive normalizado en [0,1]^2 (ver config.py) para no
    quedar calibrado a una resolucion fija -- se escala aca, una sola vez
    por camara, contra el tamano real del primer frame entregado (que puede
    diferir de cap_width/cap_height: un video de prueba conserva su
    resolucion nativa, no la de la camara real)."""
    px = roi_norm * np.array([frame_w - 1, frame_h - 1], dtype=np.float64)
    return np.round(px).astype(np.int32)


def load_models(vision_cfg: VisionConfig) -> tuple[YOLO, YOLO]:
    """Carga y fusiona los dos modelos YOLO (brazo, cajas) de vision_cfg.

    Vive fuera de run_inference a proposito: quien arranca la camara lo
    llama antes de lanzar el hilo, para que unos pesos ilegibles fallen en
    el arranque y no dejen el proceso sirviendo video sin detectar nada."""
    # Import tardio: la geometria/gate de este modulo se puede testear en
    # equipos sin Ultralytics/CUDA. El worker real si falla aqui, con el
    # mensaje de arranque que corresponde.
    from ultralytics import YOLO

    arm_model = YOLO(str(vision_cfg.arm.weights))
    arm_model.fuse()
    box_model = YOLO(str(vision_cfg.boxes.weights))
    box_model.fuse()
    return arm_model, box_model


def run_inference(cfg: PipelineConfig,
                  vision_cfg: VisionConfig,
                  drawing_cfg: DrawingConfig,
                  palletizing_cfg: PalletizingConfig,
                  isometric_cfg: IsometricConfig,
                  cam: CameraConfig,
                  arm_model: YOLO,
                  box_model: YOLO,
                  frame_q: queue.Queue,
                  jpeg_out,
                  iso_scene_out,
                  stop,
                  cam_fps_box: list[float] | None = None) -> None:
    """Consume frames de frame_q, corre los dos YOLO con model.predict(),
    entrega la observacion cruda del frame al GridCounter y dibuja lo que
    este devuelve, empujando el JPEG resultante a jpeg_out. La geometria
    del ISO se publica como SceneState para que el navegador la renderice.
    `arm_model`/`box_model` llegan ya cargados y fusionados (load_models).
    `cam_fps_box` es el out-parameter que escribe el hilo lector
    (camera_io.reader) con el fps de camara/video, para mostrarlo en el HUD
    junto al fps de este lazo (ver draw_hud). `stop` acepta threading.Event
    o multiprocessing.Event."""
    arm_cfg, box_cfg = vision_cfg.arm, vision_cfg.boxes
    tag = cam.tag

    # cam.roi esta normalizado [0,1]^2 (ver CameraConfig.roi) -- hace falta
    # el tamano real del primer frame para escalarlo a pixeles antes de
    # poder construir el GridCounter o el poligono del ROI. Ese primer
    # frame se guarda como pendiente para no perderlo: el lazo de abajo lo
    # procesa igual que a cualquier otro, no vuelve a pedirlo a frame_q.
    pending_frame = frame_q.get()
    if pending_frame is None:
        return
    frame_h, frame_w = pending_frame.shape[:2]
    roi_pts = _scale_roi(cam.roi, frame_w, frame_h)
    # El conteo NO rectifica el ROI de deteccion sino la PALETA. Son dos
    # poligonos con dos trabajos distintos, y hasta ahora se usaba uno solo
    # para los dos: roi_pts decide si un bbox esta dentro del area vigilada
    # (a proposito mas amplia, para no recortar cajas altas), mientras que la
    # homografia del cerebro tiene que llevar la SUPERFICIE DE LA PALETA al
    # cuadrado unidad -- es lo que asume toda la matematica de celdas
    # (counter.py, docs/palletizing_counting.md seccion 4). Con el ROI ahi,
    # el cuadrado unidad era un recuadro arbitrario mas grande que la tarima
    # y las distancias de palletizing.yaml no median fracciones de paleta.
    pallet_pts = _scale_roi(
        np.array(cam.pallet_corners, dtype=np.float64), frame_w, frame_h
    )

    counter = GridCounter(pallet_pts, palletizing_cfg, cam_tag=tag)
    inf_fps = 0.0
    # Ultimo render del ISO y la firma que lo produjo: se reusa mientras la
    # escena y los angulos de vista no cambien (ver el bloque del ISO abajo).
    last_confirmed_signature: tuple | None = None
    last_web_scene_signature: tuple | None = None
    pending_iso_scene: SceneState | None = None
    t_prev = time.perf_counter()
    # Reparto de tiempo del lazo, para el log periodico. Sin esto "va lento"
    # no se puede accionar: hay que saber si el costo esta en el modelo, en el
    # dibujado, en el JPEG o en esperar a la camara.
    t_log = time.perf_counter()
    n_loops = 0
    cost = {"espera": 0.0, "yolo": 0.0, "motion": 0.0, "dibujo": 0.0, "jpeg": 0.0}
    prev_gray: np.ndarray | None = None
    roi_mask: np.ndarray | None = None
    diff_buffer: np.ndarray | None = None

    rec_normal = VideoRecorder(cfg.recording, cam.id, "normal", tag)

    (arm_class_id,) = _resolve_class_ids(
        arm_model.names, (arm_cfg.class_name,), where="vision.arm.class_name",
    )
    box_class_ids = _resolve_class_ids(
        box_model.names, box_cfg.class_names, where="vision.boxes.class_names",
    )
    logger.info(
        "[%s] clases resueltas -- brazo: %r=%d (modelo %s); cajas: %s (modelo %s)",
        tag, arm_cfg.class_name, arm_class_id, arm_cfg.weights,
        ", ".join(f"{name!r}={cls_id}" for name, cls_id in zip(box_cfg.class_names, box_class_ids)),
        box_cfg.weights,
    )
    state_path = palletizing_cfg.state_directory / f"camera_{cam.id}.json"
    if not cam.persist_state:
        # Sin persistencia la paleta arranca vacia aunque exista un snapshot:
        # un archivo viejo restaurado es indistinguible de cajas reales, y en
        # una corrida de prueba eso contamina el conteo desde el frame 1.
        logger.info("[%s] persistencia de paleta desactivada (persist_state: false) -- "
                    "se ignora %s y no se escribe nada", tag, state_path)
    elif state_path.exists():
        try:
            counter.load_state(state_path)
            logger.info("[%s] estado de paleta restaurado desde %s (%d cajas)",
                        tag, state_path, counter.total)
        except ValueError as exc:
            logger.error("[%s] estado de paleta rechazado (%s): %s", tag, state_path, exc)

    # Calentamiento: la primera llamada real a model.predict() paga la
    # compilacion/alocacion de kernels CUDA (varios segundos) -- se hace
    # aca con frames dummy para que ese costo no caiga sobre el primer
    # frame de camara real ni descalibre inf_fps.
    t_warmup = time.perf_counter()
    for model, model_cfg in ((arm_model, arm_cfg), (box_model, box_cfg)):
        dummy = np.zeros((model_cfg.imgsz, model_cfg.imgsz, 3), dtype=np.uint8)
        for _ in range(3):
            model.predict(dummy, imgsz=model_cfg.imgsz, conf=model_cfg.conf,
                          device=cfg.device, verbose=False)
    logger.info("[%s] modelos calentados en %.2fs", tag, time.perf_counter() - t_warmup)

    def publish_web_scene(scene: SceneState) -> None:
        """Publica latest-value sin esperar y conserva un reintento local."""
        nonlocal last_web_scene_signature, pending_iso_scene
        signature = _web_scene_signature(scene)
        if signature != last_web_scene_signature:
            last_web_scene_signature = signature
            pending_iso_scene = scene
        if pending_iso_scene is not None and _deliver_latest(iso_scene_out, pending_iso_scene):
            pending_iso_scene = None

    try:
        while not stop.is_set():
            t_a = time.perf_counter()
            if pending_frame is not None:
                frame, pending_frame = pending_frame, None
            else:
                frame = frame_q.get()
            if frame is None:
                break
            t_b = time.perf_counter()
            cost["espera"] += t_b - t_a

            # -- Brazo primero, siempre --------------------------------------
            # Detector dedicado, 1 clase, corre en TODOS los frames.
            arm_results = arm_model.predict(
                frame,
                imgsz=arm_cfg.imgsz,
                conf=arm_cfg.conf,
                device=cfg.device,
                verbose=False,
            )[0]
            arm_bboxes = _arm_detections(
                arm_results.boxes, arm_class_id=arm_class_id, conf_floor=arm_cfg.conf,
            )
            arm_visible = bool(arm_bboxes)

            # -- Cajas solo si no hay brazo -----------------------------------
            # Si el brazo esta en escena, el frame se descarta para conteo
            # igual (ver el gate en GridCounter.update): correr el modelo de
            # cajas ahi no aporta nada y solo gasta tiempo de GPU/CPU.
            if arm_visible:
                bboxes: list[tuple[int, int, int, int, float, str]] = []
            else:
                box_results = box_model.predict(
                    frame,
                    imgsz=box_cfg.imgsz,
                    conf=box_cfg.conf,
                    device=cfg.device,
                    verbose=False,
                )[0]
                bboxes = _box_detections(
                    box_results.boxes,
                    roi_pts=roi_pts,
                    class_names=box_model.names,
                    class_conf=box_cfg.class_conf,
                    roi_tolerance_ratio=box_cfg.roi_tolerance_ratio,
                )
            t_c = time.perf_counter()
            cost["yolo"] += t_c - t_b
            # -- Observacion cruda: cuanto se movio el ROI ------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if roi_mask is None or roi_mask.shape != gray.shape:
                roi_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(roi_mask, [roi_pts], 255)
                diff_buffer = np.empty_like(gray)

            motion_score = _roi_motion_score(prev_gray, gray, roi_mask, diff_buffer)
            prev_gray = gray
            t_d = time.perf_counter()
            cost["motion"] += t_d - t_c

            # -- Actualizar FPS siempre --------------------------------------------
            now     = time.perf_counter()
            raw_fps = 1.0 / max(now - t_prev, 1e-6)
            alpha   = vision_cfg.fps_smoothing_alpha
            inf_fps = alpha * inf_fps + (1.0 - alpha) * raw_fps
            t_prev  = now

            drawing.draw_roi(frame, roi_pts, drawing_cfg)

            # Publicacion PREVIA al cerebro sin tocar el inventario. El
            # bootstrap inicial puede ejecutar una busqueda combinatoria
            # sincronica dentro de update(); si esperamos su retorno, el ISO
            # queda vacio durante todo ese calculo. Esta escena sale desde el
            # primer frame detectado con status=confirming/initializing.
            scene_before_update = counter.scene_state(isometric_cfg.visual_height_ratio)
            observed_before_update = counter.provisional_boxes(
                bboxes,
                height_ratio=isometric_cfg.visual_height_ratio,
                level_tops=scene_before_update.level_tops,
            )
            publish_web_scene(
                _with_provisional_boxes(scene_before_update, observed_before_update)
            )

            result = counter.update(FrameInput(boxes=bboxes,
                                               arm_visible=arm_visible,
                                               motion_score=motion_score))

            if result.gate is GateState.ARM_PAUSE:
                drawing.draw_arm_present(frame, roi_pts, drawing_cfg, arm_bboxes)
            elif result.gate in (GateState.MOTION_PAUSE, GateState.SETTLING):
                cv2.putText(frame, "MOVIMIENTO",
                            (roi_pts[0][0], roi_pts[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, drawing_cfg.arm_alert_font_scale,
                            drawing_cfg.color_arm_alert, drawing_cfg.arm_alert_thickness)
            else:
                drawing.draw_grid_detections(frame, result.detections, drawing_cfg)

            # Estado posterior al update: es la fuente canonica para HUD,
            # persistencia y renderer legado. La escena previa solo existio
            # para que un update costoso no dejara ciego al navegador.
            scene = counter.scene_state(isometric_cfg.visual_height_ratio)
            observed_after_update = counter.provisional_boxes(
                bboxes,
                height_ratio=isometric_cfg.visual_height_ratio,
                level_tops=scene.level_tops,
            )
            web_scene = _with_provisional_boxes(scene, observed_after_update)
            # La escena canonica (mas sus observaciones marcadas) tambien se
            # entrega antes del HUD/JPEG. Al completar min_stable, la caja
            # desaparece de provisional_boxes y aparece en boxes confirmadas.
            publish_web_scene(web_scene)
            level_counts: dict[int, int] = {}
            for box in scene.boxes:
                level_counts[box.level] = level_counts.get(box.level, 0) + 1
            box_class = scene.boxes[0].box_class if scene.boxes else ""
            cam_fps = cam_fps_box[0] if cam_fps_box else None
            drawing.draw_hud(frame, tag, counter, inf_fps, drawing_cfg,
                             box_class=box_class, level_counts=level_counts, cam_fps=cam_fps,
                             validating_initial=web_scene.validating_initial,
                             observed_count=len(web_scene.provisional_boxes))
            t_e = time.perf_counter()
            cost["dibujo"] += t_e - t_d

            # Una sola publicacion por iteracion, siempre con cajas y HUD ya
            # dibujados. Publicar tambien un preview crudo del mismo frame
            # hacia alternar ambas versiones en MJPEG, duplicaba la
            # codificacion y hacia que video/texto parecieran trabados.
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.web.jpeg_quality])
            if ok:
                push_frame(jpeg_out, buf.tobytes())
            rec_normal.write(frame)
            t_f = time.perf_counter()
            cost["jpeg"] += t_f - t_e

            # -- Vista de inspeccion 3D (seccion 10) --------------------------
            # El ISO es caro: rasteriza un canvas de ~960x960 con una decena
            # de solidos alfa-mezclados y despues lo codifica a JPEG. Hacerlo
            # en CADA frame gastaba ese costo aunque la escena fuera identica
            # a la anterior, y como corre dentro del mismo hilo que la
            # inferencia, se comia el presupuesto del stream normal: el video
            # ya marcaba una caja nueva y el ISO llegaba varios frames tarde.
            #
            # Solo cambia por dos motivos: se conto/movio una celda, o alguien
            # giro la camara. Con la escena quieta se reusa el ultimo render,
            # que es exactamente lo que se hubiera vuelto a dibujar. Ademas el
            # MJPEG solo reenvia cuando el frame cambia, asi que reencodear lo
            # mismo tampoco le servia a nadie.
            # Angulos de vista: los escribe Flask cuando alguien gira el
            # ISO con el mouse en /cam/<id>/iso.
            # La firma tiene que cubrir TODO lo que el ISO dibuja, no solo el
            # total: el desglose inicial/brazo del encabezado, que celdas hay
            # y de que tamano (el footprint se refina mientras la caja se
            # sigue viendo). Con una firma incompleta el render se quedaba
            # viejo sin que nada lo delatara -- el ISO decia "brazo:1" con el
            # HUD ya en "brazo 2".
            confirmed_signature = _confirmed_scene_signature(scene)
            if confirmed_signature != last_confirmed_signature:
                last_confirmed_signature = confirmed_signature
                if cam.persist_state and not scene.validating_initial:
                    try:
                        counter.save_state(
                            state_path, height_ratio=isometric_cfg.visual_height_ratio,
                        )
                    except OSError:
                        logger.exception("[%s] no se pudo persistir estado de paleta en %s", tag, state_path)

            # -- Log periodico de rendimiento -------------------------------
            n_loops += 1
            if cfg.runtime.fps_log_interval_s > 0:
                dt = time.perf_counter() - t_log
                if dt >= cfg.runtime.fps_log_interval_s:
                    # ms por frame de cada etapa: el reparto dice donde atacar.
                    # "espera" alto = sobra GPU y falta camara; "yolo" alto =
                    # el modelo es el techo (imgsz/modelo mas chico o TensorRT).
                    reparto = "  ".join(
                        f"{k}={1000.0 * v / n_loops:.1f}ms" for k, v in cost.items())
                    logger.info("[%s] inferencia: %.1f fps (%d frames en %.0fs) -- %s",
                                tag, n_loops / dt, n_loops, dt, reparto)
                    t_log = time.perf_counter()
                    n_loops = 0
                    for k in cost:
                        cost[k] = 0.0
    finally:
        # Ultima entrega antes de cerrar: si el render final quedo pendiente
        # (el video termino justo despues de contar una caja), sin esto el
        # visor se queda con el conteo anterior y ya no llega nada mas.
        if pending_iso_scene is not None and not _deliver_latest(iso_scene_out, pending_iso_scene):
            logger.warning("[%s] la ultima geometria del ISO no se pudo entregar", tag)
        rec_normal.close()

    logger.info("[%s] inferencia detenida", tag)
