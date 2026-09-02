from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : Captura de frames (GStreamer/NVDEC, V4L2, video de prueba)
#              aislada del resto del pipeline -- no sabe nada de YOLO ni
#              de conteo, solo entrega frames.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : open_camera() capturaba solo (RuntimeError, OSError), asi que
#              el GLib.GError de Gst.parse_launch (plugin NVIDIA ausente)
#              escapaba y se saltaba el fallback YUYV / V4L2. reader() ahora
#              loguea el traceback antes de morir -- dentro del Process
#              spawneado un hilo caido no dejaba rastro.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : log periodico de captura (runtime.fps_log_interval_s): fps
#              leidos de la camara, fps entregados a la cola y frames
#              pisados porque la inferencia no alcanzo. Sin ese ultimo dato
#              no se distingue "la camara da poco" de "la inferencia no
#              consume lo que la camara da".
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 25-08-26
# Reason     : el modo video ahora lee solo el tramo [video_start_s,
#              video_end_s) del archivo (0/0 = video completo), para
#              probar un momento puntual sin recortar el mp4 a mano. El
#              tramo es por camara: cada una tiene su propio video.
# -----------------------------------------------------------------------
"""Captura de frames: GStreamer+NVDEC / V4L2 para camaras reales, y
lectura de video de prueba para correr sin hardware. Un hilo lector por
camara (_reader) empuja frames a una queue.Queue local."""

import gc
import logging
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from boxarm.config import CameraConfig, PipelineConfig

logger = logging.getLogger(__name__)

# Apertura de dispositivo V4L2: una UVC en Jetson abre al instante pero puede
# tardar ~1s en entregar el primer frame valido, y tras un intento GStreamer
# fallido necesita un momento para soltar el descriptor.
V4L2_WARMUP_READS   = 15
V4L2_WARMUP_DELAY_S = 0.1
DEVICE_SETTLE_S     = 0.2

# -- GStreamer/NVDEC (solo modo="camara") ---------------------------------------
# gi/GStreamer se importa PEREZOSAMENTE (recien cuando se abre la primera camara
# real) para que modo="video" siga funcionando en cualquier PC de escritorio sin
# GStreamer/PyGObject instalado -- solo la Jetson necesita esta parte.
_gst_lock  = threading.Lock()
_gst_ready = False
Gst = None


def _ensure_gstreamer() -> None:
    global Gst, _gst_ready
    if _gst_ready:
        return
    with _gst_lock:
        if _gst_ready:
            return
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst as _Gst, GstApp  # noqa: F401
        _Gst.init(None)
        Gst = _Gst
        _gst_ready = True


def _sample_to_bgr(sample) -> np.ndarray | None:
    buf, caps = sample.get_buffer(), sample.get_caps()
    s = caps.get_structure(0)
    w, h = s.get_value("width"), s.get_value("height")
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        frame = np.frombuffer(mapinfo.data, dtype=np.uint8, count=h * w * 3)
        frame = frame.reshape((h, w, 3)).copy()
    finally:
        buf.unmap(mapinfo)
    return frame


class GstCamera:
    """Wrapper de un pipeline GStreamer con appsink -- misma interfaz minima
    que cv2.VideoCapture (read()/release()) para que _reader_camara no tenga
    que distinguir entre GStreamer y el fallback V4L2."""

    def __init__(self, pipeline_str: str, timeout_s: float = 2.0) -> None:
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._sink = self._pipeline.get_by_name("sink")
        self._timeout = int(timeout_s * Gst.SECOND)
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            # Soltar la referencia ademas de parar el pipeline: si solo se para,
            # el objeto sigue vivo dentro del traceback de esta excepcion y
            # v4l2src mantiene abierto /dev/videoN, con lo que el fallback V4L2
            # se encuentra el dispositivo ocupado (EBUSY -> isOpened() False).
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            self._sink = None
            raise RuntimeError("pipeline no llego a PLAYING")

    def read(self):
        sample = self._sink.try_pull_sample(self._timeout)
        if sample is None:
            return False, None
        frame = _sample_to_bgr(sample)
        return (frame is not None), frame

    def release(self) -> None:
        self._pipeline.set_state(Gst.State.NULL)


def _pipeline_mjpeg_nvdec(cam_idx: int, w: int, h: int, fps: int) -> str:
    return (
        "v4l2src device=/dev/video{idx} ! "
        "image/jpeg,width={w},height={h},framerate={fps}/1 ! "
        "nvv4l2decoder mjpeg=1 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
    ).format(idx=cam_idx, w=w, h=h, fps=fps)


def _pipeline_yuyv(cam_idx: int, w: int, h: int, fps: int) -> str:
    return (
        "v4l2src device=/dev/video{idx} ! "
        "video/x-raw,format=YUY2,width={w},height={h},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
    ).format(idx=cam_idx, w=w, h=h, fps=fps)


def open_camera(cam_idx: int, tag: str, cap_width: int, cap_height: int, fps_request: int,
                 gst_timeout_s: float):
    """Abre /dev/video{cam_idx} intentando, en orden: GStreamer+NVDEC (MJPEG
    con decodificacion por hardware), GStreamer con YUYV crudo, y por ultimo
    V4L2 plano via cv2.VideoCapture. Devuelve el primer objeto que entregue
    un frame de prueba real, o None si los 3 caminos fallan -- mismo patron
    de open_camera() en pipeline_unificado_vm_editable.py."""
    _ensure_gstreamer()

    candidates = [
        ("MJPEG + NVDEC", lambda: _pipeline_mjpeg_nvdec(cam_idx, cap_width, cap_height, fps_request)),
        ("YUYV crudo",    lambda: _pipeline_yuyv(cam_idx, cap_width, cap_height, fps_request)),
    ]
    for label, build in candidates:
        logger.info("[%s] probando: %s (%dx%d)", tag, label, cap_width, cap_height)
        cam = None
        try:
            cam = GstCamera(build(), timeout_s=gst_timeout_s)
            ok, frame = cam.read()
            if ok and frame is not None:
                logger.info("[%s] OK -- usando: %s", tag, label)
                return cam
            raise RuntimeError("no se obtuvo frame de prueba")
        except Exception as exc:  # noqa: BLE001 -- Gst.parse_launch lanza GLib.GError
            # Amplio a propositio: si falta nvv4l2decoder/nvvidconv (Jetson sin
            # plugins NVIDIA), parse_launch lanza gi.repository.GLib.GError, que
            # no es RuntimeError ni OSError. Capturarlo estrecho hacia que la
            # excepcion escapara y se saltara el fallback YUYV / V4L2 plano.
            logger.warning("[%s] %s fallo: %s", tag, label, exc)
            if cam is not None:
                cam.release()
        # Forzar el ciclo de GC: mientras el pipeline muerto siga vivo, v4l2src
        # no cierra /dev/videoN y el siguiente candidato (o el fallback V4L2) se
        # lo encuentra ocupado. Solo en el camino de fallo -- el de exito ya
        # devolvio la camara arriba.
        cam = None
        gc.collect()
        time.sleep(DEVICE_SETTLE_S)

    logger.warning("[%s] GStreamer nativo fallo -- usando V4L2 plano", tag)
    return _open_v4l2(cam_idx, tag, cap_width, cap_height, fps_request)


def _open_v4l2(cam_idx: int, tag: str, cap_width: int, cap_height: int,
                fps_request: int):
    """Fallback V4L2 plano via cv2.VideoCapture. No basta con isOpened(): una
    UVC en Jetson abre al instante pero tarda en entregar el primer frame, asi
    que se descartan lecturas vacias hasta V4L2_WARMUP_READS antes de dar la
    camara por buena -- devolver el cap sin comprobarlo hacia que el lector
    entrara en un ciclo de reconexion perpetuo."""
    cap = cv2.VideoCapture(f"/dev/video{cam_idx}", cv2.CAP_V4L2)
    if not cap.isOpened():
        logger.warning("[%s] V4L2 no pudo abrir /dev/video%d (ocupado o sin permisos)", tag, cam_idx)
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv2.CAP_PROP_FPS, fps_request)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(V4L2_WARMUP_READS):
        try:
            ok, frame = cap.read()
        except cv2.error:
            ok, frame = False, None
        if ok and frame is not None and frame.size > 0:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info("[%s] OK -- usando: V4L2 plano (%dx%d)", tag, w, h)
            return cap
        time.sleep(V4L2_WARMUP_DELAY_S)

    logger.warning("[%s] V4L2 abrio /dev/video%d pero no entrega frames", tag, cam_idx)
    cap.release()
    return None


def push_frame(frame_q: queue.Queue, frame: np.ndarray) -> bool:
    """Empuja un frame descartando el mas viejo si la cola esta llena --
    prioriza siempre el frame mas reciente sobre no perder ninguno (video
    en vivo o de prueba, no un procesamiento offline).

    Devuelve False si aun asi no entro. Quien empuja UN frame por cambio de
    estado (y no uno por frame) tiene que reintentar: para el, un descarte
    no se corrige solo en la iteracion siguiente, se pierde hasta el proximo
    cambio."""
    try:
        frame_q.put_nowait(frame)
        return True
    except queue.Full:
        try:
            frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            frame_q.put_nowait(frame)
            return True
        except queue.Full:
            return False


# Peso del fps anterior en el promedio movil de cam_fps_box -- mismo valor
# que VisionConfig.fps_smoothing_alpha usa para inf_fps en inference.py, asi
# las dos lineas del HUD ("FPS camara"/"FPS infer") suavizan igual y son
# comparables entre si en vez de que una salte mas que la otra por usar
# ventanas distintas.
_CAM_FPS_SMOOTHING_ALPHA = 0.85


def reader(cam: CameraConfig, cfg: PipelineConfig, subsample_factor: int,
           frame_q: queue.Queue, stop, cam_fps_box: list[float] | None = None) -> None:
    """Despacha al lector de camara en vivo o de video de prueba segun
    cfg.modo -- mismo patron que thread_acquire() en
    pipeline_unificado_vm_editable.py. `stop` acepta threading.Event o
    multiprocessing.Event (misma interfaz: is_set/set/wait).
    `subsample_factor` viene de VisionConfig (configs/vision.yaml).

    `cam_fps_box` es un list[float] de 1 elemento usado como out-parameter:
    este hilo escribe ahi el fps de lectura (suavizado), y run_inference lo
    lee para el HUD sin necesitar una cola/lock -- un solo escritor, un solo
    lector, un float no necesita mas que eso. None si no hace falta reportarlo
    (p.ej. en tests).

    Cualquier excepcion se loguea con traceback antes de propagar: este hilo
    corre dentro de un multiprocessing.Process spawneado, donde un fallo sin
    loguear muere en silencio y la camara solo "no conecta"."""
    try:
        if cfg.modo == "video":
            _reader_video(cam.video, cam.tag, cfg.loop_video, cam.video_start_s,
                          cam.video_end_s, cam.video_speed, subsample_factor, cfg.runtime,
                          frame_q, stop, cam_fps_box)
        else:
            _reader_camara(cam.index, cam.tag, cfg, subsample_factor, frame_q, stop, cam_fps_box)
    except Exception:  # noqa: BLE001 -- ultimo recurso: loguear antes de morir
        logger.exception("[%s] lector abortado por excepcion no controlada", cam.tag)
        frame_q.put(None)
        raise


def _reader_camara(cam_index: int,
                    tag: str,
                    cfg: PipelineConfig,
                    subsample_factor: int,
                    frame_q: queue.Queue,
                    stop,
                    cam_fps_box: list[float] | None = None) -> None:
    """Abre /dev/video{cam_index} via open_camera() (GStreamer+NVDEC, con
    fallback a V4L2 plano) y empuja frames a frame_q (submuestreo segun
    subsample_factor, igual que detect_video_1.py). Si la camara falla o
    nunca llega a abrir, reconecta sola (cfg.runtime.reconnect_delay_s) en
    vez de matar el hilo -- mismo patron de reconexion que
    _thread_acquire_camara en pipeline_unificado_vm_editable.py."""
    rt = cfg.runtime
    idx = 0
    cap = None
    missing_device_logged = False
    # Contadores del log periodico de captura. `dropped` son los frames que
    # el lector empujo y la inferencia nunca vio porque la cola estaba llena:
    # es la medida directa de cuanto se esta quedando atras la inferencia
    # respecto de la camara.
    t_log = time.perf_counter()
    n_read = n_pushed = n_dropped = 0
    t_prev_frame: float | None = None
    cam_fps_ema = 0.0

    while not stop.is_set():
        if cap is None:
            dev = f"/dev/video{cam_index}"
            if not os.path.exists(dev):
                if not missing_device_logged:
                    logger.error(
                        "[%s] no existe %s; modo camara requiere un dispositivo Linux. "
                        "Para probar el video en Windows use modo: video.",
                        tag, dev,
                    )
                    missing_device_logged = True
                stop.wait(rt.reconnect_delay_s)
                continue
            missing_device_logged = False
            cap = open_camera(cam_index, tag, cfg.cap_width, cfg.cap_height, cfg.fps_request, rt.gst_timeout_s)
            if cap is None:
                logger.warning("[%s] no se pudo abrir %s, reintentando en %ss...", tag, dev, rt.reconnect_delay_s)
                time.sleep(rt.reconnect_delay_s)
                continue

        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning("[%s] fallo de lectura, reconectando...", tag)
            cap.release()
            cap = None
            time.sleep(rt.read_fail_delay_s)
            continue

        now = time.perf_counter()
        if t_prev_frame is not None:
            raw_fps = 1.0 / max(now - t_prev_frame, 1e-6)
            cam_fps_ema = _CAM_FPS_SMOOTHING_ALPHA * cam_fps_ema + (1.0 - _CAM_FPS_SMOOTHING_ALPHA) * raw_fps
            if cam_fps_box is not None:
                cam_fps_box[0] = cam_fps_ema
        t_prev_frame = now

        idx += 1
        n_read += 1
        if idx % subsample_factor == 0:
            # push_frame descarta el frame viejo cuando la cola esta llena y
            # aun asi devuelve True, asi que el descarte hay que verlo ANTES:
            # cola llena = la inferencia todavia no consumio el anterior.
            if frame_q.full():
                n_dropped += 1
            if push_frame(frame_q, frame):
                n_pushed += 1

        if rt.fps_log_interval_s > 0:
            dt = time.perf_counter() - t_log
            if dt >= rt.fps_log_interval_s:
                logger.info("[%s] captura: %.1f fps leidos de la camara, %.1f fps a la cola "
                            "(submuestreo 1/%d) -- %d frame(s) pisados porque la "
                            "inferencia no alcanzo",
                            tag, n_read / dt, n_pushed / dt, subsample_factor, n_dropped)
                t_log = time.perf_counter()
                n_read = n_pushed = n_dropped = 0

    if cap is not None:
        cap.release()
    frame_q.put(None)
    logger.info("[%s] lector (camara) detenido", tag)


def _reader_video(video_path: Path,
                   tag: str,
                   loop_video: bool,
                   start_s: float,
                   end_s: float,
                   video_speed: float,
                   subsample_factor: int,
                   runtime,
                   frame_q: queue.Queue,
                   stop,
                   cam_fps_box: list[float] | None = None) -> None:
    """Lee el video de prueba de esta camara, respetando su FPS original
    (para que la comprobacion se vea a velocidad real) y aplicando el mismo
    submuestreo que el modo camara. Solo entrega el tramo
    [start_s, end_s) del archivo; end_s = 0 significa "hasta el final".
    Si loop_video esta activo, repite ese mismo tramo al terminar -- mismo
    patron que _thread_acquire_video en pipeline_unificado_vm_editable.py.

    `video_speed` (CameraConfig.video_speed) acelera/frena el sleep de
    abajo sin tocar el archivo ni el modelo: 2.0 entrega el doble de rapido
    que en vivo, para no esperar los minutos reales de un video de prueba
    largo. 1.0 = tiempo real (default)."""
    idx = 0
    tramo = f"{start_s:.1f}s -> {'fin' if end_s <= 0 else f'{end_s:.1f}s'}"
    logger.info("[%s] modo VIDEO  path=%s  tramo=%s  loop=%s  velocidad=%.1fx",
                tag, video_path, tramo, loop_video, video_speed)

    while not stop.is_set():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("[%s] no se pudo abrir el video %s", tag, video_path)
            time.sleep(runtime.reconnect_delay_s)
            if not loop_video:
                break
            continue

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay = (1.0 / fps) / video_speed
        # Ritmo pautado a proposito (el sleep de abajo lo fuerza): el fps
        # efectivo de "camara" en este modo es el del video de origen
        # escalado por video_speed, no hace falta medirlo por delta de
        # tiempo como en _reader_camara.
        if cam_fps_box is not None:
            cam_fps_box[0] = fps * video_speed

        # El salto se pide en milisegundos: con contenedores de fps variable
        # el salto por numero de frame cae en otro instante del que dice el
        # yaml. Si el backend no soporta el seek, POS_MSEC se queda en 0 y
        # el corte por end_s de abajo igual respeta la duracion pedida.
        if start_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

        while not stop.is_set():
            t0 = time.perf_counter()
            if end_s > 0 and cap.get(cv2.CAP_PROP_POS_MSEC) >= end_s * 1000.0:
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            idx += 1
            if idx % subsample_factor == 0:
                push_frame(frame_q, frame)

            wait = delay - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)

        cap.release()
        if not loop_video or stop.is_set():
            break

    frame_q.put(None)
    logger.info("[%s] lector (video) detenido", tag)
