from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm -- Changelog
# Programmer  | Date     | Resumen
# ----------- | -------- | -----------------------------------------------
# gerald      | 23-08-26 | Entry point de multiprocessing por camara: hilo
#             |          | lector + hilo de inferencia en un proceso propio.
# gerald      | 23-08-26 | Recibe VisionConfig y DrawingConfig por separado
#             |          | en vez de solo PipelineConfig.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : Cola de geometria JSON para la vista web
#              de inspeccion 3D (/cam/<id>/iso, seccion 10 del
#              documento) -- cruza en paralelo al jpeg_q normal.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : el modulo de vision paso a llamarse inference y su entry
#              point a run_inference (tracking.py/run_tracker quedaron
#              obsoletos tras mover las decisiones a palletizing).
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : el modelo se carga aqui, antes de arrancar los hilos: unos
#              pesos ilegibles ahora abortan el proceso-camara (y el padre
#              decide si relanzarlo) en vez de matar solo al hilo de
#              inferencia y dejar el stream corriendo sin detectar nada.
# -----------------------------------------------------------------------
"""camera_worker() es el target de multiprocessing.Process para una
camara: corre en su propio interprete, con su propio hilo lector y su
propio hilo de inferencia YOLO -- si el proceso muere, no afecta a las
demas camaras."""

import faulthandler
import logging
import os
import queue
import threading

from boxarm.capture import camera_io
from boxarm.runtime.logging_config import configure_logging
from boxarm.vision import inference
from boxarm.config import (
    CameraConfig, DrawingConfig, IsometricConfig, PalletizingConfig, PipelineConfig, VisionConfig,
)

logger = logging.getLogger(__name__)


def camera_worker(cam: CameraConfig,
                   cfg: PipelineConfig,
                   vision_cfg: VisionConfig,
                   drawing_cfg: DrawingConfig,
                   palletizing_cfg: PalletizingConfig,
                   isometric_cfg: IsometricConfig,
                   jpeg_q, iso_scene_q, stop) -> None:
    """Target de multiprocessing.Process para UNA camara. `jpeg_q` (stream
    normal) y `iso_scene_q` (geometria para el renderer local del navegador) son
    multiprocessing.Queue que cruzan al proceso
    principal; `stop` es un multiprocessing.Event compartido por todas
    las camaras."""
    configure_logging()
    # BOXARM_STALL_DUMP=1: cada 20s vuelca la pila de TODOS los hilos de
    # este proceso a stderr (stdlib puro, sin instalar nada). Sirve para
    # ver la linea EXACTA donde esta pegado un hilo con 90%+ de CPU sin
    # ninguna excepcion ni log -- adivinar por comentarios de codigo no
    # alcanza cuando el bloqueo puede estar en cualquiera de varias
    # funciones candidatas. Apagado por default: es ruido en una corrida
    # normal.
    if os.environ.get("BOXARM_STALL_DUMP") == "1":
        faulthandler.enable()
        faulthandler.dump_traceback_later(20, repeat=True)

    # Antes que los hilos: si los pesos no cargan (version de ultralytics
    # incompatible, archivo ausente), el proceso termina aqui con el motivo
    # a la vista. Dentro del hilo esto moria en un traceback suelto y la
    # camara seguia sirviendo video sin inferencia.
    try:
        arm_model, box_model = inference.load_models(vision_cfg)
    except Exception:
        logger.exception(
            "[%s] no se pudo cargar el modelo de brazo (%s) o de cajas (%s) -- la camara no arranca",
            cam.tag, vision_cfg.arm.weights, vision_cfg.boxes.weights,
        )
        return

    # Latest-frame semantics: if inference falls behind, an older frame is
    # less valuable than the newest one.  A single-slot queue prevents stale
    # frames from adding latency while retaining the existing drop policy in
    # camera_io.push_frame().
    frame_q = queue.Queue(maxsize=1)
    # Out-parameter de 1 solo elemento: el hilo lector escribe el fps de
    # camara/video ahi, el de inferencia lo lee para el HUD. Un solo
    # escritor/lector dentro del MISMO proceso -- no hace falta lock ni
    # multiprocessing.Value (ver docstring de camera_io.reader).
    cam_fps_box = [0.0]

    t_read = threading.Thread(
        target=camera_io.reader,
        args=(cam, cfg, vision_cfg.subsample_factor, frame_q, stop, cam_fps_box),
        name=f"reader-{cam.tag}",
    )
    t_infer = threading.Thread(
        target=inference.run_inference,
        args=(cfg, vision_cfg, drawing_cfg, palletizing_cfg, isometric_cfg, cam,
              arm_model, box_model, frame_q, jpeg_q, iso_scene_q, stop,
              cam_fps_box),
        name=f"inference-{cam.tag}",
    )
    t_read.start()
    t_infer.start()

    # El Ctrl+C del proceso padre llega tambien aqui como KeyboardInterrupt:
    # sin capturarlo, el join() de arriba lo propagaba como traceback y el
    # proceso moria a medio apagar en vez de dejar terminar a sus dos hilos.
    try:
        t_read.join()
        t_infer.join()
    except KeyboardInterrupt:
        stop.set()
        t_read.join(timeout=cfg.runtime.shutdown_join_timeout_s)
        t_infer.join(timeout=cfg.runtime.shutdown_join_timeout_s)

    logger.info("[%s] proceso de camara detenido", cam.tag)
