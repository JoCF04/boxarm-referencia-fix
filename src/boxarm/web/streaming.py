from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm -- Changelog
# Programmer  | Date     | Resumen
# ----------- | -------- | -----------------------------------------------
# gerald      | 23-08-26 | UN solo Flask/puerto sirviendo N camaras por ruta
#             |          | /cam/<id>, en vez de una app y un puerto por camara.
# gerald      | 23-08-26 | mjpeg_poll_s y drain_timeout_s desde
#             |          | PipelineConfig.runtime en vez de hardcodeados.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : (1) el buffer JPEG se generaliza a un _JpegStore reusable,
#              para poder servir un SEGUNDO stream por camara (la vista
#              isometrica de docs/palletizing_counting.md seccion 10) en
#              /cam/<id>/iso, en paralelo al stream normal /cam/<id>.
#              (2) _mjpeg_gen() reenviaba el mismo frame en loop cerrado
#              cuando no llegaba uno nuevo (sin esperar mjpeg_poll_s) --
#              saturaba la conexion con frames duplicados. Ahora siempre
#              espera mjpeg_poll_s entre frames enviados.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : el consumidor de los angulos del ISO pasa a llamarse
#              run_inference (vision/inference.py).
# -----------------------------------------------------------------------
"""Streaming HTTP y panel unificado de cada camara.

Los hilos drain_* llenan los buffers JPEG; Flask presenta el video real y
el pallet digital juntos en /cam/<id>.
"""

import logging
import math
import queue
import threading
import time
from dataclasses import asdict

import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from boxarm.config import CameraConfig, DrawingConfig, IsometricConfig, RuntimeConfig
from boxarm.vision.palletizing.formulas import _project, build_homography

logger = logging.getLogger(__name__)


class _JpegStore:
    """Buffer JPEG compartido para UN tipo de stream (normal o iso),
    con un solo lock para todas las camaras -- igual que _jpeg_lock en
    pipeline_unificado_vm_editable.py. Vive en el proceso principal: cada
    camara corre en su propio proceso aparte y llega hasta aqui via
    multiprocessing.Queue (ver drain())."""

    def __init__(self, name: str = "store", trace: bool = False) -> None:
        # Nombre solo para los logs -- hay dos instancias (normal e iso) y sin
        # esto no se distingue cual de las dos se quedo sin frames.
        self._name = name
        # El stream normal empuja un frame por frame: trazar cada uno seria
        # ruido puro. El ISO empuja solo cuando cambia el conteo o la vista,
        # asi que sus pocas lineas se pueden mirar en INFO y sirven para
        # ubicar en que salto se pierde un frame (cola -> drain -> cliente).
        self._trace_level = logging.INFO if trace else logging.DEBUG
        self._lock = threading.Lock()
        self._latest: dict[int, bytes | None] = {}
        # Version por camara: se incrementa en cada write. Permite a un
        # cliente MJPEG distinguir "frame nuevo" de "el mismo de antes"
        # sin comparar los bytes del JPEG.
        self._seq: dict[int, int] = {}

    def register(self, cam_id: int) -> None:
        self._latest.setdefault(cam_id, None)
        self._seq.setdefault(cam_id, 0)

    def write(self, cam_id: int, data: bytes | None) -> None:
        with self._lock:
            self._latest[cam_id] = data
            self._seq[cam_id] = self._seq.get(cam_id, 0) + 1

    def read(self, cam_id: int) -> tuple[bytes | None, int]:
        with self._lock:
            return self._latest.get(cam_id), self._seq.get(cam_id, 0)

    def drain(self, cam_id: int, jpeg_q, runtime: RuntimeConfig, stop) -> None:
        """Hilo en el proceso principal, uno por camara: recibe los
        frames ya codificados a JPEG que empuja el proceso camera_worker
        de esa camara (via jpeg_q) y los deja en este buffer -- puente
        entre el proceso de la camara y el servidor Flask."""
        while not stop.is_set():
            try:
                data = jpeg_q.get(timeout=runtime.drain_timeout_s)
            except queue.Empty:
                continue
            self.write(cam_id, data)
            logger.log(self._trace_level,
                       "[cam %d] %s: frame recibido del proceso camara (%s bytes) -> seq %d",
                       cam_id, self._name, "None" if data is None else len(data),
                       self._seq.get(cam_id, 0))

    def mjpeg_gen(self, cam_id: int, mjpeg_poll_s: float,
                  max_fps: float | None = None):
        """Generador MJPEG -- solo lee el buffer, no hace inferencia
        (igual que _mjpeg_gen en pipeline_unificado_vm_editable.py).

        Envia un frame SOLO cuando cambio la version: antes se reenviaba
        el ultimo JPEG cada mjpeg_poll_s aunque la camara no hubiera
        producido nada nuevo, asi que a 0.02 s de poll salian 50 frames/s
        por cliente sobre una fuente de ~15 fps -- el grueso del trafico
        eran duplicados que el navegador ya estaba mostrando.

        `max_fps` limita UNICAMENTE lo enviado a este cliente HTTP. El
        productor, la cola latest-frame, YOLO y el conteo conservan su ritmo;
        si llegan varios JPEG durante el intervalo se envia el ultimo."""
        last_seq = -1
        last_sent_at: float | None = None
        min_interval = 0.0 if max_fps is None else 1.0 / max_fps
        while True:
            data, seq = self.read(cam_id)
            now = time.monotonic()
            due = last_sent_at is None or now - last_sent_at >= min_interval
            if data is not None and seq != last_seq and due:
                logger.log(self._trace_level,
                           "[cam %d] %s: enviando seq %d al cliente MJPEG", cam_id, self._name, seq)
                last_seq = seq
                last_sent_at = now
                # Content-Length es obligatorio aca, no un adorno. En un
                # multipart/x-mixed-replace sin largo declarado, el navegador
                # solo sabe que una parte termino cuando ve el separador de
                # la SIGUIENTE -- asi que el ultimo frame enviado se queda sin
                # pintar hasta que llegue otro. Con el stream normal no se
                # notaba (manda un frame por frame, cada uno empuja al
                # anterior a la pantalla), pero el ISO solo manda cuando
                # cambia el conteo: su ultimo frame quedaba invisible por
                # segundos, o para siempre si el video ya termino.
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(data)).encode("ascii") + b"\r\n"
                       b"\r\n" + data + b"\r\n")
            time.sleep(mjpeg_poll_s)


normal_store = _JpegStore("normal")


class _SceneStore:
    """Ultima geometria ISO por camara.

    A diferencia del JPEG, este dato pesa pocos kilobytes y solo cambia al
    contar o refinar una caja. El navegador lo proyecta localmente durante
    el arrastre, sin pedir un frame nuevo a Python por cada pixel del mouse.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[int, object | None] = {}
        self._seq: dict[int, int] = {}

    def register(self, cam_id: int) -> None:
        self._latest.setdefault(cam_id, None)
        self._seq.setdefault(cam_id, 0)

    def write(self, cam_id: int, scene) -> None:
        with self._lock:
            self._latest[cam_id] = scene
            self._seq[cam_id] = self._seq.get(cam_id, 0) + 1

    def read(self, cam_id: int) -> tuple[object | None, int]:
        with self._lock:
            return self._latest.get(cam_id), self._seq.get(cam_id, 0)

    def drain(self, cam_id: int, scene_q, runtime: RuntimeConfig, stop) -> None:
        while not stop.is_set():
            try:
                scene = scene_q.get(timeout=runtime.drain_timeout_s)
            except queue.Empty:
                continue
            self.write(cam_id, scene)


scene_store = _SceneStore()

def drain_jpeg_queue(cam_id: int, jpeg_q, runtime: RuntimeConfig, stop) -> None:
    """Stream normal (frame anotado por GridCounter)."""
    normal_store.drain(cam_id, jpeg_q, runtime, stop)


def drain_iso_scene_queue(cam_id: int, scene_q, runtime: RuntimeConfig, stop) -> None:
    """Geometria compacta para el renderer interactivo del navegador."""
    scene_store.drain(cam_id, scene_q, runtime, stop)


def _rgb(color: tuple[int, int, int]) -> list[int]:
    """Convierte el BGR de OpenCV al RGB de CSS/Canvas."""
    return [int(color[2]), int(color[1]), int(color[0])]


def make_flask_app(
    cameras: tuple[CameraConfig, ...],
    runtime: RuntimeConfig,
    default_view: tuple[float, float] = (35.0, 35.0),
    isometric_cfg: IsometricConfig | None = None,
    drawing_cfg: DrawingConfig | None = None,
    stream_max_fps: float | None = None,
) -> Flask:
    """UN solo Flask, un solo puerto, sirviendo las N camaras por ruta
    /cam/<id> como panel unificado (video + ISO) -- reemplaza las N apps con N puertos de
    la version anterior. `default_view` es (azimuth, elevation) de
    configs/isometric.yaml: la vista a la que vuelve el visor al resetear."""
    # El HTML/CSS/JS del visor vive en templates/ y static/, no incrustado
    # aca: mezclar una pagina entera dentro de un .py deja el markup sin
    # resaltado, sin cache del navegador y imposible de editar aparte.
    app = Flask(__name__, template_folder="templates", static_folder="static")
    # Jinja cachea la plantilla compilada y solo revisa el mtime si `debug`
    # esta activo -- que aca no lo esta. Sin esto, editar camera.html no tiene
    # efecto hasta reiniciar el proceso, mientras que iso.js SI se relee en
    # cada request (es un estatico): el JS nuevo se queda buscando nodos que
    # el HTML viejo todavia no tiene y el visor revienta con un null. El
    # costo es un stat por render, despreciable frente a un visor de video.
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    # Flask sirve static/ con max-age de 12h. Combinado con el auto-reload de
    # arriba eso deja el visor a medio actualizar: el HTML llega nuevo y el
    # CSS/JS sigue siendo el de hace horas -- y dentro de los iframes del
    # dashboard ni un recargado duro del navegador los renueva. El visor
    # corre en LAN contra un proceso local; no hay ancho de banda que
    # justifique cachear cuatro archivos de texto.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    tags_by_id = {cam.id: cam.tag for cam in cameras}
    enabled_by_id = {cam.id: bool(getattr(cam, "enabled", True)) for cam in cameras}
    # Misma homografia que build_homography() usa en inference.py (via
    # GridCounter): las esquinas de la PALETA en fraccion [0,1] del frame,
    # no el ROI de deteccion ni pixeles --
    # la transformacion es invariante a la unidad. Se cachea por camara para
    # proyectar las esquinas del pallet exactamente como se proyecta cada
    # bbox de caja, en vez de asumir un cuadrado unidad de antemano.
    homography_by_id = {
        cam.id: build_homography(np.array(cam.pallet_corners, dtype=np.float64))
        for cam in cameras
        if getattr(cam, "pallet_corners", None)
    }
    # Esquinas del deck por camara: cada una calibra su propia paleta en
    # configs/roi_cam_<id>.json (pallet_roi), no hay una global.
    pallet_corners_by_id = {
        cam.id: tuple(getattr(cam, "pallet_corners", ()) or ())
        for cam in cameras
    }
    # Log de arranque: mismo formato que las cajas para poder comparar a
    # ojo -- esquinas crudas (fraccion de frame) -> proyectadas ([0,1]^2
    # del render). Si el pallet no calza con las cajas en el iso, esto
    # dice de entrada si el problema esta en la proyeccion o en otro lado.
    for cam in cameras:
        homography = homography_by_id.get(cam.id)
        corners = pallet_corners_by_id.get(cam.id)
        if homography is None or not corners:
            continue
        projected = [_project(homography, x, y) for x, y in corners]
        logger.info(
            "[Camara %d] pallet corners fraccion=%s -> proyectado[0,1]^2=%s",
            cam.id,
            [tuple(round(v, 4) for v in pt) for pt in corners],
            [tuple(round(v, 4) for v in pt) for pt in projected],
        )
    for cam_id in tags_by_id:
        normal_store.register(cam_id)
        scene_store.register(cam_id)

    @app.route("/cam/<int:cam_id>")
    def route_cam(cam_id: int):
        if cam_id not in tags_by_id:
            return f"camara {cam_id} no existe", 404
        az0 = request.args.get("az", default=default_view[0], type=float)
        el0 = request.args.get("el", default=default_view[1], type=float)
        if not math.isfinite(az0):
            az0 = default_view[0]
        if not math.isfinite(el0):
            el0 = default_view[1]
        el0 = max(-89.0, min(89.0, el0))
        view_mode = request.args.get("view", "inspection")
        if view_mode not in {"iso", "camera", "inspection"}:
            view_mode = "inspection"
        return render_template("camera.html", cam_id=cam_id, tag=tags_by_id[cam_id],
                               az0=az0, el0=el0, view_mode=view_mode,
                               camera_enabled=enabled_by_id[cam_id])

    @app.route("/cam/<int:cam_id>/stream")
    def route_cam_stream(cam_id: int):
        """MJPEG crudo que alimenta la ventana de video del panel."""
        if cam_id not in tags_by_id:
            return f"camara {cam_id} no existe", 404
        return Response(normal_store.mjpeg_gen(
                            cam_id, runtime.mjpeg_poll_s, stream_max_fps),
                         mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/cam/<int:cam_id>/snapshot")
    def route_cam_snapshot(cam_id: int):
        """Ultimo JPEG finito para el capturador Chromium.

        El navegador humano usa MJPEG; Chromium refresca este recurso antes
        de cada screenshot para que la captura no espere un stream infinito.
        """
        if cam_id not in tags_by_id:
            return f"camara {cam_id} no existe", 404
        data, _ = normal_store.read(cam_id)
        if data is None:
            response = Response("camara aun sin frame", status=503)
        else:
            response = Response(data, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/cam/<int:cam_id>/iso/scene")
    def route_cam_iso_scene(cam_id: int):
        """Snapshot JSON latest-only para el renderer local del visor.

        Los angulos no pasan por este endpoint: son estado puramente visual
        del navegador y se actualizan a la frecuencia de la pantalla.
        """
        if cam_id not in tags_by_id:
            return f"camara {cam_id} no existe", 404
        scene, seq = scene_store.read(cam_id)
        view = None
        colors = None
        if isometric_cfg is not None:
            view = {
                "fill_margin": isometric_cfg.fill_margin,
                "level_gap_ratio": isometric_cfg.level_gap_ratio,
            }
            # Compatibilidad con consumidores antiguos del contrato ISO; el
            # renderer actual usa la geometría proyectada del pallet cuando
            # está disponible.
            if hasattr(isometric_cfg, "pallet_width_m"):
                view["pallet_width"] = isometric_cfg.pallet_width_m
            if hasattr(isometric_cfg, "pallet_length_m"):
                view["pallet_length"] = isometric_cfg.pallet_length_m
            if drawing_cfg is not None and pallet_corners_by_id.get(cam_id):
                # Mismo proceso que una caja: box.u/box.v salen de proyectar
                # su bbox (fraccion de frame) con la homografia de ESTA
                # camara (scene.py: _project(self._homography, cx, cy)).
                # Las esquinas del pallet (pallet_roi de
                # configs/roi_cam_<id>.json, misma fraccion [0,1] de frame
                # que el main_roi) se proyectan IGUAL, con la homografia
                # real de cam_id.
                #
                # OJO: esto NO se manda como "pallet_width"/"pallet_length"
                # para reescalar la posicion de la caja (box.u * pallet_width)
                # -- esa era la atadura real: las cajas viven siempre en
                # [0,1]^2 puro, sin multiplicar por nada del pallet. Lo que
                # se manda es el RECTANGULO propio del pallet (x0,x1,y0,y1)
                # en ESE MISMO [0,1]^2, para que el renderer lo dibuje al
                # lado de las cajas sin tocar su escala.
                homography = homography_by_id.get(cam_id)
                corners = pallet_corners_by_id[cam_id]
                if homography is not None:
                    projected = [_project(homography, x, y) for x, y in corners]
                else:
                    projected = list(corners)
                xs = [p[0] for p in projected]
                ys = [p[1] for p in projected]
                view["pallet_x0"] = min(xs)
                view["pallet_x1"] = max(xs)
                view["pallet_y0"] = min(ys)
                view["pallet_y1"] = max(ys)
                view["pallet_visible"] = drawing_cfg.pallet_visible
                # Fraccion del tamano REAL del pallet (no metros: el mundo
                # de la escena es [0,1]^2 sin unidad fisica) -- iso.js las
                # aplica sobre floorSize = min(ancho, alto) del pallet_roi
                # proyectado, no sobre el dominio [0,1] completo de la caja.
                view["pallet_deck_thickness"] = drawing_cfg.pallet_deck_thickness_m
                view["pallet_support_height"] = drawing_cfg.pallet_support_height_m
        if drawing_cfg is not None:
            colors = {
                "background": _rgb(drawing_cfg.hud_background),
                "title": _rgb(drawing_cfg.color_hud_title),
                "pallet": _rgb(drawing_cfg.color_roi),
                "classes": {
                    name: list(rgb)
                    for name, rgb in getattr(isometric_cfg, "class_colors", {}).items()
                } if isometric_cfg is not None else {},
            }
        response = jsonify({
            "seq": seq,
            "scene": asdict(scene) if scene is not None else None,
            "view": view,
            "colors": colors,
        })
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/api/cameras")
    def route_cameras_status():
        """Estado compacto de todas las cámaras, incluidas las deshabilitadas."""
        cameras_payload = []
        for cam_id, tag in tags_by_id.items():
            _jpeg, jpeg_seq = normal_store.read(cam_id)
            scene, scene_seq = scene_store.read(cam_id)
            if not enabled_by_id[cam_id]:
                status = "disabled"
            elif jpeg_seq > 0 or scene_seq > 0:
                status = "online"
            else:
                status = "no_signal"
            scene_data = asdict(scene) if scene is not None else None
            cameras_payload.append({
                "id": cam_id,
                "tag": tag,
                "enabled": enabled_by_id[cam_id],
                "status": status,
                "stream_seq": jpeg_seq,
                "scene_seq": scene_seq,
                "scene": scene_data,
            })
        # Misma paleta RGB que manda /iso/scene en colors.classes: el bullet
        # de cada clase en el dashboard tiene que ser el color con el que el
        # renderer pinta esa caja, no uno propio. Es global (no depende de
        # cam_id), asi que viaja una vez al lado de la lista.
        class_colors = {
            name: list(rgb)
            for name, rgb in getattr(isometric_cfg, "class_colors", {}).items()
        } if isometric_cfg is not None else {}
        response = jsonify({"cameras": cameras_payload, "class_colors": class_colors})
        response.headers["Cache-Control"] = "no-store"
        return response


    @app.route("/")
    def route_home():
        return render_template("dashboard.html", cameras=[
            {"id": cam_id, "tag": tag, "enabled": enabled_by_id[cam_id]}
            for cam_id, tag in tags_by_id.items()
        ])

    return app
