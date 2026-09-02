from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : Configuracion tipada del pipeline, cargada desde
#              configs/pipeline.yaml -- ver G-5 (nada hardcodeado).
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : Identificadores de config a ingles (niveles->levels_layout,
#              umbrales->thresholds, tau_celda->tau_cell,
#              tau_solape_celda->tau_cell_overlap, deteccion->detection),
#              las 8 constantes que vivian hardcodeadas en
#              grid_counting.py pasan a configs/palletizing.yaml (G-5), y
#              el gate de pausa (movimiento + brazo) se mueve a
#              PalletizingConfig.gate porque es logica de negocio del
#              cerebro de paletizado, no de la inferencia.
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 25-08-26
# Reason     : video_start_s/video_end_s por camara (viven en cada entrada de
#              "camaras", junto a su video), validados aqui:
#              un rango invalido dejaba al lector sin frames y eso se veia
#              como "la camara no conecta".
# -----------------------------------------------------------------------
"""Configuracion tipada del pipeline (P-9): PipelineConfig, CameraConfig,
ArmDetectionConfig, BoxDetectionConfig, cargados y validados una sola vez
desde YAML."""

import json
import math

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class ArmDetectionConfig:
    """Detector dedicado del brazo robotico -- ver configs/vision.yaml.

    Modelo aparte del de cajas a proposito (ver comentario en
    configs/vision.yaml): 1 sola clase, presencia simple, entrenado con
    TODAS las fotos (incluidas las de brazo tapando/con motion blur, que
    es justo el caso que mas importa acertar)."""
    weights: Path
    imgsz: int
    conf: float
    class_name: str


@dataclass(frozen=True)
class BoxDetectionConfig:
    """Detector de cajas -- ver configs/vision.yaml.

    Las clases se declaran por NOMBRE, no por id. El id de una clase depende
    del orden del `names:` del dataset y se corre al agregar etiquetas
    nuevas; el nombre no. Configurar por id dejaba una clase apuntando a
    otra sin ningun error visible (el id seguia existiendo, solo que ahora
    era otra clase)."""
    weights: Path
    imgsz: int
    conf: float
    class_conf: dict[str, float]
    class_names: tuple[str, ...]
    roi_tolerance_ratio: float = 0.05


@dataclass(frozen=True)
class RuntimeConfig:
    """Timeouts y reconexion de captura/streaming/apagado -- ver
    configs/pipeline.yaml. La deteccion vive en VisionConfig
    (configs/vision.yaml)."""
    reconnect_delay_s: float
    read_fail_delay_s: float
    gst_timeout_s: float
    mjpeg_poll_s: float
    drain_timeout_s: float
    shutdown_join_timeout_s: float
    main_loop_tick_s: float
    camera_restart_delay_s: float
    max_camera_restarts: int
    fps_log_interval_s: float  # 0 = sin logs periodicos de rendimiento


@dataclass(frozen=True)
class RecordingTypesConfig:
    normal: bool
    iso: bool
    dashboard: bool


@dataclass(frozen=True)
class DashboardRecordingConfig:
    capture_host: str
    azimuth_deg: float
    elevation_deg: float
    width: int
    height: int
    jpeg_quality: int
    startup_timeout_s: float


@dataclass(frozen=True)
class RecordingConfig:
    """Grabacion opcional por tipo bajo output_dir/cam/<id>."""
    enabled: bool
    output_dir: Path  # relativo a la raiz del repo
    fps: float          # fps de reproduccion del archivo grabado, no del stream en vivo
    fourcc: str           # codec de cv2.VideoWriter_fourcc, p.ej. "mp4v"
    extension: str          # extension del archivo, p.ej. "mp4"
    types: RecordingTypesConfig
    dashboard: DashboardRecordingConfig
    # cv2.VideoWriter con "mp4v" (MPEG-4 Part 2) no lo reproducen WhatsApp
    # ni la mayoria de apps de chat -- piden H.264 en MP4. OpenCV en Windows
    # no trae encoder H.264 (falta openh264.dll, con licencia aparte), asi
    # que se graba con mp4v igual y, al cerrar el archivo, se reencodea a
    # H.264 con ffmpeg (subprocess, no libreria) si esta disponible. Si
    # ffmpeg no esta instalado, se deja el mp4v tal cual (se loguea una vez).
    transcode_h264: bool
    transcode_timeout_s: float

    def type_enabled(self, kind: str) -> bool:
        """Combina el interruptor global con el del tipo solicitado."""
        if kind not in ("normal", "iso", "dashboard"):
            raise ValueError(f"tipo de grabacion desconocido: {kind}")
        return self.enabled and bool(getattr(self.types, kind))


@dataclass(frozen=True)
class VisionConfig:
    """Deteccion frame-a-frame -- ver configs/vision.yaml.

    No contiene tracker: la identidad persistente pertenece al estado de
    ocupacion de la paleta, no a trayectorias temporales del detector.
    """
    arm: ArmDetectionConfig
    boxes: BoxDetectionConfig
    fps_smoothing_alpha: float
    subsample_factor: int  # procesar 1 de cada N frames leidos


@dataclass(frozen=True)
class LevelLayout:
    """Celdas de UN nivel: centros `(u, v)` en el cuadrado unidad
    [0,1]^2, medidos de una imagen de calibracion -- NO una rejilla
    filas x columnas. El arreglo real no es necesariamente regular (las
    cajas pueden ir en distintas orientaciones formando un patron tipo
    rompecabezas, no una cuadricula), asi que cada celda se declara por
    su posicion, no se calcula dividiendo el plano en partes iguales. El
    indice de LevelLayout en PalletizingConfig.levels_layout ES el nivel
    `z`; el indice de cada celda en `cells` ES su identidad `g`."""
    cells: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class GateConfig:
    """Cuando el cerebro tiene permiso para contar -- ver el bloque `gate:`
    de configs/palletizing.yaml. No es config del tracker: decide si un
    frame se valida o se descarta, que es una decision de paletizado."""
    motion_pause_enabled: bool    # si True, no valida cajas mientras hay movimiento dentro del ROI
    motion_diff_threshold: float  # cambio promedio de gris dentro del ROI para declarar movimiento
    motion_stable_frames: int     # frames quietos consecutivos antes de reanudar validacion
    arm_debounce_frames: int      # frames consecutivos sin brazo antes de declarar cerrado el ciclo
    # Frames COUNTING consecutivos sin ninguna deteccion, habiendo cajas ya
    # confirmadas, antes de asumir que la paleta se vacio en la realidad y
    # resetear el conteo (GridCounter.reset_pallet(), ver frame_loop.py). El
    # mismo orden de magnitud que motion_stable_frames: una oclusion
    # momentanea no debe leerse como un vaciado real.
    empty_pallet_debounce_frames: int


@dataclass(frozen=True)
class ConfirmationConfig:
    """Confirmacion temporal corta de una candidata, antes de persistirla.

    No es tracking: al confirmar, la identidad pasa a ser ``(celda, nivel)``.
    """
    min_stable: int       # frames COUNTING consecutivos antes de confirmar una caja nueva
    same_box_iou: float   # IoU minimo para enlazar la misma candidata entre esos frames


@dataclass(frozen=True)
class PalletizingConfig:
    """Conteo por rejilla y escala aparente -- ver
    docs/palletizing_counting.md y configs/palletizing.yaml. Los 4 vertices
    de la paleta no se repiten aqui: son CameraConfig.roi, calibrados en
    configs/roi_cam_<id>.json (misma geometria que ya filtra detecciones).
    Nada de dimensiones de render ni angulos de vista aqui -- eso es
    IsometricConfig (configs/isometric.yaml), un concern puramente
    visual que no afecta el conteo."""
    reference_scale_px: float  # max(w,h) esperado de una caja completa en el nivel 0
    c_z: float                  # altura del centro optico sobre la paleta (m)
    box_height: float            # estimacion de calibracion para la escalera de niveles; no sale del bbox 2D
    layout_mode: str             # "auto" descubre posiciones; "fixed" usa levels_layout[].cells
    state_directory: Path        # JSON persistente por camara/paleta actual
    occupancy_grid: int              # resolucion del mapa de ocupacion del cuadrado unidad
    levels_layout: tuple[LevelLayout, ...]  # plantillas fijas; en auto no limita los niveles absolutos
    gate: GateConfig                   # cuando se permite contar (movimiento + brazo)
    confirmation: ConfirmationConfig   # estabilidad previa de una candidata nueva
    tau_rung: float                    # tolerancia relativa para asignar la escala a un peldano
    tau_rec: float                      # margen relativo bajo el peldano mas bajo -> recorte, no nivel
    tau_cell: float                      # distancia maxima (en [0,1]^2) a la celda mas cercana -> F2
    tau_overlap: float                   # solape minimo para tratar una deteccion debil como caja ocluida debajo
    tau_overlap_center: float            # distancia maxima normalizada entre centroides para la heuristica de oclusion
    tau_cell_overlap: float              # solape de footprint sobre el menor con una celda ya descubierta -> es la MISMA caja, no una nueva
    min_stack_area_ratio: float          # area minima relativa a la celda que pisa para aceptar una caja apilada
    min_complete_side_ratio: float       # lado minimo contra la mediana para no declarar recorte
    max_same_level_overlap: float        # interpenetracion maxima antes de rechazar una caja nueva
    overlap_warn_ratio: float             # piso de ruido del marcador visual/log SOLAPE -- solo diagnostico
    partial_fit_tolerance: float         # holgura al reconstruir una caja desde un recorte (etiquetado + precision del robot)
    max_duplicate_scale_ratio: float     # relacion de tamano por encima de la cual dos bboxes solapados son LA MISMA caja
    free_gap_ratio: float                # fraccion del footprint de consenso que debe medir un hueco para que "quepa una caja"
    min_support_coverage: float          # fraccion del footprint que debe sostener la union de celdas de abajo
    max_support_ratio: float             # maximo s1/s2 entre los dos soportes dominantes; 2 permite hasta 60/30
    max_bootstrap_combinations: int      # limite de seguridad al producto cartesiano de hipotesis de reconciliacion inicial
    max_position_correction: float       # desplazamiento maximo al corregir el centro de una celda ya confirmada

    @property
    def levels(self) -> int:
        if self.layout_mode == "fixed":
            return len(self.levels_layout)

        # No existe un maximo operativo arbitrario en modo auto. Este numero
        # expresa solamente el dominio fisico de la calibracion proyectiva:
        # la cara superior de una caja debe permanecer por debajo de c_z.
        # c_z > n * box_height  =>  n_max = ceil(c_z / box_height) - 1.
        return max(1, int(np.ceil(self.c_z / self.box_height)) - 1)


@dataclass(frozen=True)
class IsometricConfig:
    """Vista de inspeccion 3D (docs/palletizing_counting.md seccion 10)
    -- ver configs/isometric.yaml. Solo diagnostico visual, no afecta el
    conteo (eso es PalletizingConfig). El footprint en X,Y ("a","b") NO
    se configura aqui: se
    MIDE dinamicamente desde los bbox detectados (GridCounter.footprint(),
    via la homografia). Como las cajas son iguales, el ISO normaliza la
    carga con las medianas de lado largo/corto y conserva la orientacion.

    La escala de pantalla (px/m) tampoco se fija aqui: antes era un valor
    manual (scale_px_per_m) que quedaba chico o grande segun el pallet
    real, dejando la escena diminuta en el canvas. Ahora se AUTO-CALCULA
    en cada frame para que la escena (pallet + torre completa de niveles)
    llene `fill_margin` del canvas -- ver el renderer local del navegador (`iso.js`).

    El alto fisico de caja ("c", eje Z) no puede obtenerse del bbox de una
    sola camara. Por eso el render NO presenta PalletizingConfig.box_height
    como una medicion: usa una extrusion puramente visual. El alto dibujado
    es `visual_height_ratio * min(box_a, box_b)`.

    La geometria de la paleta (tamano, soportes, si se dibuja) NO vive aqui
    -- es un rectangulo de fondo fijo, ajeno al ROI/homografia, y por eso
    esta en DrawingConfig/drawing.yaml junto al resto del "como se ve"."""
    azimuth_deg: float        # theta -- rotacion horizontal de la vista
    elevation_deg: float       # phi -- inclinacion de la vista
    canvas_width: int
    canvas_height: int
    fill_margin: float          # fraccion del canvas (0-1) que debe ocupar la escena proyectada
    visual_height_ratio: float   # extrusion esquematica; NO es la altura fisica Z
    level_gap_ratio: float        # separacion vertical DIBUJADA entre niveles, en fracciones del alto de un nivel (vista explotada)
    class_colors: dict[str, tuple[int, int, int]] = field(default_factory=dict)


Color = Tuple[int, int, int]


@dataclass(frozen=True)
class DrawingConfig:
    """Colores (BGR) y layout de las anotaciones -- ver configs/drawing.yaml."""
    color_new: Color
    color_redet: Color
    color_pending: Color
    color_roi: Color
    color_text: Color
    color_hud_title: Color
    color_arm_alert: Color
    level_colors: tuple[Color, ...]

    roi_thickness: int

    box_thickness: int
    box_label_font_scale: float
    box_label_thickness: int
    box_circle_radius: int

    arm_alert_font_scale: float
    arm_alert_thickness: int

    hud_width: int
    hud_height: int
    hud_background: Color
    hud_title_font_scale: float
    hud_title_thickness: int
    hud_line_font_scale: float
    hud_line_thickness: int
    hud_visible_thickness: int
    hud_fps_font_scale: float
    hud_fps_thickness: int

    # Aspecto de la paleta 3D (vista /cam/<id>/iso). Las 4 esquinas NO estan
    # aca: son geometria calibrada por camara y viven en
    # CameraConfig.pallet_corners (configs/roi_cam_<id>.json). Aca queda solo
    # lo que es "como se ve" y es igual para todas: visibilidad y grosores.
    pallet_visible: bool
    pallet_deck_thickness_m: float
    pallet_board_gap_m: float
    pallet_support_height_m: float
    pallet_support_width_m: float
    pallet_floor_margin_ratio: float


@dataclass(frozen=True)
class CameraConfig:
    id: int      # numero de camara para la ruta HTTP /cam/<id> -- posicion (1-based) en la lista del YAML
    index: int   # numero de /dev/videoN, solo modo="camara"
    tag: str
    video: Path
    # 4 vertices [u, v] en el cuadrado unidad [0,1]^2, NO en pixeles: la
    # resolucion real de una camara (o de un video de prueba) puede diferir
    # de cap_width/cap_height -- un ROI en pixeles fijos quedaba calibrado
    # para una resolucion y desencajaba en cualquier otra. Se escala a
    # pixeles una vez, al arrancar cada camara, contra el tamano real del
    # primer frame (ver _scale_roi en inference.py).
    #
    # NO sale de pipeline.yaml: es "main_roi.normalized" de
    # configs/roi_cam_<id>.json, el archivo que escribe
    # scripts/calibrate_roi.py. Un solo lugar donde vive la calibracion, en
    # vez de copiar a mano los mismos numeros al YAML.
    roi: np.ndarray = field(repr=False)
    # Esquinas REALES del deck de la paleta -- "pallet_roi.normalized" del
    # mismo roi_cam_<id>.json. Mismo formato y mismo frame que `roi` (que a
    # proposito es mas amplio, para no recortar cajas altas). Es por camara
    # porque cada camara ve su propia paleta: antes era un unico
    # drawing.pallet.corners global, que solo podia estar bien para una.
    pallet_corners: tuple[tuple[float, float], ...] = field(default=())
    video_start_s: float = 0.0   # solo modo video -- segundo donde arranca (0 = desde el inicio)
    video_end_s: float = 0.0     # solo modo video -- segundo donde corta (0 = hasta el final)
    # solo modo video -- multiplicador del ritmo de entrega. 1.0 = tiempo
    # real (el sleep de _reader_video pausa al fps propio del archivo, para
    # que una prueba se vea como se veria en vivo). >1.0 acelera esa espera
    # SIN tocar el video de origen ni el modelo -- sirve para no esperar los
    # 12 minutos reales de un video de prueba largo. No tiene efecto en
    # modo="camara" (ahi no hay sleep artificial, el fps lo pone el sensor).
    video_speed: float = 1.0
    enabled: bool = True  # false -> no arranca proceso ni ruta /cam/<id> para esta camara
    # false -> la paleta de esta camara arranca siempre vacia: ni se lee el
    # snapshot existente ni se escribe uno nuevo. Es por camara porque una
    # puede estar contando una paleta real mientras otra reproduce un video
    # de prueba, y el estado de esa prueba no debe sobrevivir a la corrida.
    persist_state: bool = True


@dataclass(frozen=True)
class WebConfig:
    """Servidor/transporte HTTP (configs/web.yaml): separado de
    PipelineConfig para no mezclar "como se sirve" con "como se captura y
    detecta"."""
    flask_host: str
    port: int
    jpeg_quality: int
    stream_max_fps: float


@dataclass(frozen=True)
class PipelineConfig:
    modo: str
    loop_video: bool
    device: str
    fps_request: int
    cap_width: int
    cap_height: int
    web: WebConfig
    runtime: RuntimeConfig
    recording: RecordingConfig
    cameras: tuple[CameraConfig, ...]


class ConfigError(Exception):
    """Configuracion de pipeline invalida o incompleta."""


def _resolve_path(base: Path, raw: str) -> Path:
    """Rutas relativas a la raiz del repo, nunca absolutas hardcodeadas (G-6)."""
    path = Path(raw)
    return path if path.is_absolute() else (base / path)


def load_web_config(config_path: Path) -> WebConfig:
    """Carga y valida configs/web.yaml: host/puerto/calidad JPEG del
    servidor Flask, sin nada de captura ni deteccion."""
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"no se pudo leer {config_path}") from exc

    try:
        # OpenCV clampea silenciosamente fuera de 0..100: un valor como 140
        # se codifica a calidad 100 (frames ~5x mas pesados) sin avisar.
        jpeg_quality = int(raw["jpeg_quality"])
        if not 0 <= jpeg_quality <= 100:
            raise ConfigError(
                f"jpeg_quality={jpeg_quality} fuera de rango en {config_path}: debe estar entre 0 y 100"
            )
        stream_max_fps = float(raw["stream_max_fps"])
        if not math.isfinite(stream_max_fps) or stream_max_fps <= 0:
            raise ConfigError(
                f"stream_max_fps={stream_max_fps} invalido en {config_path}: debe ser mayor que 0"
            )
        return WebConfig(
            flask_host=str(raw["flask_host"]),
            port=int(raw["port"]),
            jpeg_quality=jpeg_quality,
            stream_max_fps=stream_max_fps,
        )
    except KeyError as exc:
        raise ConfigError(f"falta la clave {exc} en {config_path}") from exc


def load_pipeline_config(config_path: Path) -> PipelineConfig:
    """Carga y valida configs/pipeline.yaml una sola vez; el resto del
    codigo consume PipelineConfig ya tipado, sin volver a validar (P-9).
    El servidor web es un concern aparte: se lee de web.yaml, siempre
    hermano de pipeline.yaml dentro de configs/."""
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"no se pudo leer {config_path}") from exc

    web = load_web_config(config_path.parent / "web.yaml")

    base = config_path.resolve().parent.parent  # raiz del repo, config vive en configs/

    try:
        def _roi(raw_roi, *, where: str) -> np.ndarray:
            """4 vertices [u, v] normalizados en [0,1]^2. Rechaza en el
            arranque un ROI fuera de rango en vez de dejarlo escalar a un
            poligono que cae fuera del frame real (ver CameraConfig.roi)."""
            # roi_cam_<id>.json guarda [{"x": .., "y": ..}, ...]; el
            # calibrador escribe ese formato porque es el que se lee a ojo
            # al revisar el archivo. Se acepta tal cual en vez de obligar a
            # un paso de conversion aparte.
            if raw_roi and isinstance(raw_roi[0], dict):
                raw_roi = [[point["x"], point["y"]] for point in raw_roi]
            arr = np.array(raw_roi, dtype=np.float64)
            if arr.shape != (4, 2):
                raise ConfigError(f"{where} debe tener 4 vertices [u, v] en {config_path}")
            if not np.all((arr >= 0.0) & (arr <= 1.0)):
                raise ConfigError(
                    f"{where} debe estar normalizado en [0, 1] en {config_path}: {raw_roi!r}"
                )
            return _order_corners(arr)

        def _order_corners(arr: np.ndarray) -> np.ndarray:
            """Reordena los 4 vertices a sup-izq, sup-der, inf-der, inf-izq.

            build_homography() (formulas.py) asume EXACTAMENTE ese orden: es
            el que mapea a [(0,0), (1,0), (1,1), (0,1)]. calibrate_roi.py, en
            cambio, guarda los puntos en el orden en que se clickearon --
            empezar por la esquina inferior derecha es tan valido para el
            operador como empezar por la superior izquierda, pero deja la
            homografia rotada/espejada y las cajas proyectadas fuera del
            cuadrado unidad (o sea, invisibles en el iso).

            Criterio: x+y es minimo en sup-izq y maximo en inf-der; de los
            dos restantes, el de menor y es sup-der. Vale para cualquier
            cuadrilatero convexo sin lados casi verticales invertidos, que es
            lo que es un ROI de paleta.
            """
            order = np.argsort(arr[:, 0] + arr[:, 1])
            top_left = arr[order[0]]
            bottom_right = arr[order[-1]]
            rest = arr[[order[1], order[2]]]
            if rest[0][1] <= rest[1][1]:
                top_right, bottom_left = rest[0], rest[1]
            else:
                top_right, bottom_left = rest[1], rest[0]
            return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float64)

        rt_raw = raw["runtime"]
        runtime = RuntimeConfig(
            reconnect_delay_s=float(rt_raw["reconnect_delay_s"]),
            read_fail_delay_s=float(rt_raw["read_fail_delay_s"]),
            gst_timeout_s=float(rt_raw["gst_timeout_s"]),
            mjpeg_poll_s=float(rt_raw["mjpeg_poll_s"]),
            drain_timeout_s=float(rt_raw["drain_timeout_s"]),
            shutdown_join_timeout_s=float(rt_raw["shutdown_join_timeout_s"]),
            main_loop_tick_s=float(rt_raw["main_loop_tick_s"]),
            camera_restart_delay_s=float(rt_raw["camera_restart_delay_s"]),
            max_camera_restarts=int(rt_raw["max_camera_restarts"]),
            fps_log_interval_s=float(rt_raw["fps_log_interval_s"]),
        )

        rec_raw = raw["recording"]
        rec_types = rec_raw["types"]
        dashboard_raw = rec_raw["dashboard"]
        recording = RecordingConfig(
            enabled=bool(rec_raw["enabled"]),
            output_dir=_resolve_path(base, rec_raw["output_dir"]),
            fps=float(rec_raw["fps"]),
            fourcc=str(rec_raw["fourcc"]),
            extension=str(rec_raw["extension"]),
            types=RecordingTypesConfig(
                normal=bool(rec_types["normal"]),
                iso=bool(rec_types["iso"]),
                dashboard=bool(rec_types["dashboard"]),
            ),
            dashboard=DashboardRecordingConfig(
                capture_host=str(dashboard_raw["capture_host"]),
                azimuth_deg=float(dashboard_raw["azimuth_deg"]),
                elevation_deg=float(dashboard_raw["elevation_deg"]),
                width=int(dashboard_raw["width"]),
                height=int(dashboard_raw["height"]),
                jpeg_quality=int(dashboard_raw["jpeg_quality"]),
                startup_timeout_s=float(dashboard_raw["startup_timeout_s"]),
            ),
            transcode_h264=bool(rec_raw.get("transcode_h264", True)),
            transcode_timeout_s=float(rec_raw["transcode_timeout_s"]),
        )
        if recording.fps <= 0:
            raise ConfigError(f"recording.fps debe ser > 0 en {config_path}")
        if recording.transcode_timeout_s <= 0:
            raise ConfigError(f"recording.transcode_timeout_s debe ser > 0 en {config_path}")
        if recording.dashboard.width <= 0 or recording.dashboard.height <= 0:
            raise ConfigError(f"recording.dashboard width/height deben ser > 0 en {config_path}")
        if not 0 <= recording.dashboard.jpeg_quality <= 100:
            raise ConfigError(
                f"recording.dashboard.jpeg_quality debe estar entre 0 y 100 en {config_path}"
            )
        if recording.dashboard.capture_host.strip() in ("", "0.0.0.0", "::", "[::]"):
            raise ConfigError(
                f"recording.dashboard.capture_host debe ser un host navegable en {config_path}"
            )
        if not np.isfinite(recording.dashboard.azimuth_deg):
            raise ConfigError(f"recording.dashboard.azimuth_deg debe ser finito en {config_path}")
        if not -89.0 <= recording.dashboard.elevation_deg <= 89.0:
            raise ConfigError(
                f"recording.dashboard.elevation_deg debe estar entre -89 y 89 en {config_path}"
            )
        if recording.dashboard.startup_timeout_s <= 0:
            raise ConfigError(f"recording.dashboard.startup_timeout_s debe ser > 0 en {config_path}")

        def _camera_roi_file(cam_id: int) -> dict:
            """Los dos poligonos calibrados de una camara.

            Viven en configs/roi_cam_<id>.json (lo escribe
            scripts/calibrate_roi.py) y no en pipeline.yaml: son geometria
            medida sobre una imagen concreta, no algo que se edite a mano.
            Tenerlos en un solo lugar evita el caso en que el YAML y el JSON
            discrepan y nadie sabe cual gana.
            """
            roi_path = config_path.parent / f"roi_cam_{cam_id}.json"
            try:
                raw_roi = json.loads(roi_path.read_text(encoding="utf-8"))
            except OSError as exc:
                raise ConfigError(
                    f"falta {roi_path} para la camara {cam_id}: genera el ROI con "
                    f"scripts/calibrate_roi.py --camera {cam_id}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{roi_path} no es JSON valido: {exc}") from exc
            for key in ("main_roi", "pallet_roi"):
                if key not in raw_roi or "normalized" not in raw_roi[key]:
                    raise ConfigError(f"falta {key}.normalized en {roi_path}")
            return raw_roi

        def _camera(cam_id: int, cam: dict) -> CameraConfig:
            """Una entrada de la lista "camaras" ya validada. El recorte
            temporal (video_start_s/video_end_s, 0/0 = video completo) se
            valida aqui y no en el lector porque un rango invalido deja al
            lector sin ningun frame que entregar y eso se ve como "la camara
            no conecta"."""
            start_s = float(cam.get("video_start_s", 0.0))
            end_s = float(cam.get("video_end_s", 0.0))
            if start_s < 0 or end_s < 0:
                raise ConfigError(
                    f"video_start_s/video_end_s negativos en la camara {cam['tag']!r} "
                    f"de {config_path}: deben ser >= 0"
                )
            if end_s > 0 and end_s <= start_s:
                raise ConfigError(
                    f"video_end_s={end_s} <= video_start_s={start_s} en la camara "
                    f"{cam['tag']!r} de {config_path}: el fin debe ser posterior al "
                    "inicio (0 = hasta el final del video)"
                )
            video_speed = float(cam.get("video_speed", 1.0))
            if video_speed <= 0:
                raise ConfigError(
                    f"video_speed={video_speed} en la camara {cam['tag']!r} de "
                    f"{config_path}: debe ser > 0 (1.0 = tiempo real)"
                )
            roi_raw = _camera_roi_file(cam_id)
            roi_path = config_path.parent / f"roi_cam_{cam_id}.json"
            pallet = _roi(
                roi_raw["pallet_roi"]["normalized"], where=f"pallet_roi de {roi_path.name}"
            )
            return CameraConfig(
                id=cam_id,   # posicion 1-based en la lista -- es el <id> de /cam/<id>
                index=int(cam["index"]),
                tag=str(cam["tag"]),
                video=_resolve_path(base, cam["video"]),
                roi=_roi(
                    roi_raw["main_roi"]["normalized"], where=f"main_roi de {roi_path.name}"
                ),
                pallet_corners=tuple((float(x), float(y)) for x, y in pallet),
                video_start_s=start_s,
                video_end_s=end_s,
                video_speed=video_speed,
                enabled=bool(cam.get("enabled", True)),
                persist_state=bool(cam.get("persist_state", True)),
            )

        cameras = tuple(
            _camera(cam_id, cam) for cam_id, cam in enumerate(raw["camaras"], start=1)
        )

        return PipelineConfig(
            modo=str(raw["modo"]),
            loop_video=bool(raw["loop_video"]),
            device=str(raw["device"]),
            fps_request=int(raw["fps_request"]),
            cap_width=int(raw["cap_width"]),
            cap_height=int(raw["cap_height"]),
            web=web,
            runtime=runtime,
            recording=recording,
            cameras=cameras,
        )
    except KeyError as exc:
        raise ConfigError(f"falta la clave {exc} en {config_path}") from exc


def load_vision_config(config_path: Path) -> VisionConfig:
    """Carga y valida configs/vision.yaml: deteccion sin tracking temporal."""
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"no se pudo leer {config_path}") from exc

    base = config_path.resolve().parent.parent  # raiz del repo, config vive en configs/

    def _conf(value: float, where: str) -> float:
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"{where} debe estar entre 0 y 1 en {config_path}")
        return value

    try:
        for obsolete, replacement in (
            ("detection", "arm/boxes"),
        ):
            if obsolete in raw:
                raise ConfigError(
                    f"{obsolete} ya no se acepta en {config_path}: la deteccion "
                    f"se separo en dos modelos, usar {replacement} (ver el "
                    f"comentario al inicio del archivo)"
                )

        arm_raw = raw["arm"]
        arm = ArmDetectionConfig(
            weights=_resolve_path(base, arm_raw["weights"]),
            imgsz=int(arm_raw["imgsz"]),
            conf=_conf(float(arm_raw["conf"]), "arm.conf"),
            class_name=str(arm_raw["class_name"]),
        )

        box_raw = raw["boxes"]
        class_conf = {
            str(class_name): _conf(float(value), "boxes.class_conf")
            for class_name, value in (box_raw.get("class_conf") or {}).items()
        }
        box_class_names = tuple(str(value) for value in box_raw["class_names"])
        if not box_class_names:
            raise ConfigError(
                f"boxes.class_names esta vacio en {config_path} -- "
                f"sin clases de caja no hay nada que contar"
            )
        boxes = BoxDetectionConfig(
            weights=_resolve_path(base, box_raw["weights"]),
            imgsz=int(box_raw["imgsz"]),
            conf=_conf(float(box_raw["conf"]), "boxes.conf"),
            class_conf=class_conf,
            class_names=box_class_names,
            roi_tolerance_ratio=float(box_raw.get("roi_tolerance_ratio", 0.05)),
        )

        return VisionConfig(
            arm=arm,
            boxes=boxes,
            fps_smoothing_alpha=float(raw["fps_smoothing_alpha"]),
            subsample_factor=int(raw["subsample_factor"]),
        )
    except KeyError as exc:
        raise ConfigError(f"falta la clave {exc} en {config_path}") from exc


def load_palletizing_config(config_path: Path) -> PalletizingConfig:
    """Carga y valida configs/palletizing.yaml una sola vez -- solo se
    consume por GridCounter (docs/palletizing_counting.md)."""
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"no se pudo leer {config_path}") from exc

    try:
        cam_raw = raw["camera"]
        levels_raw = raw.get("levels_layout") or []
        tau_raw = raw["thresholds"]
        gate_raw = raw.get("gate") or {}
        confirmation_raw = raw.get("confirmation") or {}

        layout_mode = str(raw.get("layout_mode", "auto")).lower()
        if layout_mode == "fixed" and not levels_raw:
            raise ConfigError(
                f"'levels_layout' esta vacio en {config_path} -- se necesita al menos un nivel"
            )

        levels_layout = tuple(
            LevelLayout(cells=tuple((float(c[0]), float(c[1])) for c in lvl["cells"]))
            for lvl in levels_raw
        )
        for z, layout in enumerate(levels_layout):
            if not layout.cells:
                raise ConfigError(f"el nivel {z} no tiene celdas en {config_path}")

        c_z = float(cam_raw["c_z"])
        box_height = float(cam_raw["box_height"])
        state_directory = Path(raw.get("state_directory", "state/pallets"))
        if not state_directory.is_absolute():
            state_directory = (config_path.parent.parent / state_directory).resolve()

        if layout_mode not in {"auto", "fixed"}:
            raise ConfigError(
                f"layout_mode debe ser 'auto' o 'fixed' en {config_path}, no {layout_mode!r}"
            )
        calibrated_levels = (
            len(levels_layout)
            if layout_mode == "fixed"
            else max(1, int(np.ceil(c_z / box_height)) - 1)
        )
        if c_z <= calibrated_levels * box_height:
            raise ConfigError(
                f"c_z ({c_z}) debe ser mayor que levels*box_height "
                f"({calibrated_levels * box_height}) en {config_path} -- si no, s(z) diverge "
                f"o se vuelve negativo"
            )

        occupancy_grid = int(raw.get("occupancy_grid", 200))
        if occupancy_grid < 1:
            raise ConfigError(
                f"occupancy_grid debe ser >= 1 en {config_path}, no {occupancy_grid}"
            )

        min_support_coverage = float(tau_raw.get("min_support_coverage", 0.75))
        if not np.isfinite(min_support_coverage) or not 0.0 <= min_support_coverage <= 1.0:
            raise ConfigError(
                f"thresholds.min_support_coverage debe estar entre 0 y 1 en "
                f"{config_path}, no {min_support_coverage}"
            )
        max_support_ratio = float(tau_raw.get("max_support_ratio", 2.0))
        if not np.isfinite(max_support_ratio) or max_support_ratio < 1.0:
            raise ConfigError(
                f"thresholds.max_support_ratio debe ser >= 1 en {config_path}, "
                f"no {max_support_ratio}"
            )

        max_bootstrap_combinations = int(raw.get("max_bootstrap_combinations", 20_000))
        if max_bootstrap_combinations < 1:
            raise ConfigError(
                f"max_bootstrap_combinations debe ser >= 1 en {config_path}, "
                f"no {max_bootstrap_combinations}"
            )

        tau_cell = float(tau_raw["tau_cell"])
        max_position_correction = float(
            tau_raw.get("max_position_correction", tau_cell / 5.0)
        )
        if not 0.0 < max_position_correction <= 1.0:
            raise ConfigError(
                f"thresholds.max_position_correction debe estar en (0, 1] en "
                f"{config_path}, no {max_position_correction}"
            )

        empty_pallet_debounce_frames = int(gate_raw.get("empty_pallet_debounce_frames", 3))
        if empty_pallet_debounce_frames < 1:
            raise ConfigError(
                f"gate.empty_pallet_debounce_frames debe ser >= 1 en {config_path}, "
                f"no {empty_pallet_debounce_frames}"
            )
        gate = GateConfig(
            motion_pause_enabled=bool(gate_raw.get("motion_pause_enabled", True)),
            motion_diff_threshold=float(gate_raw.get("motion_diff_threshold", 6.0)),
            motion_stable_frames=int(gate_raw.get("motion_stable_frames", 2)),
            arm_debounce_frames=int(gate_raw.get("arm_debounce_frames", 3)),
            empty_pallet_debounce_frames=empty_pallet_debounce_frames,
        )
        min_stable = int(confirmation_raw.get("min_stable", 3))
        same_box_iou = float(confirmation_raw.get("same_box_iou", 0.25))
        if min_stable < 1:
            raise ConfigError(f"confirmation.min_stable debe ser >= 1 en {config_path}")
        if not 0.0 <= same_box_iou <= 1.0:
            raise ConfigError(
                f"confirmation.same_box_iou debe estar entre 0 y 1 en {config_path}"
            )
        confirmation = ConfirmationConfig(
            min_stable=min_stable,
            same_box_iou=same_box_iou,
        )

        return PalletizingConfig(
            reference_scale_px=float(cam_raw["reference_scale_px"]),
            c_z=c_z,
            box_height=box_height,
            layout_mode=layout_mode,
            state_directory=state_directory,
            occupancy_grid=occupancy_grid,
            levels_layout=levels_layout,
            gate=gate,
            confirmation=confirmation,
            tau_rung=float(tau_raw["tau_rung"]),
            tau_rec=float(tau_raw["tau_rec"]),
            tau_cell=tau_cell,
            tau_overlap=float(tau_raw.get("tau_overlap", 0.40)),
            tau_overlap_center=float(tau_raw.get("tau_overlap_center", 0.60)),
            tau_cell_overlap=float(tau_raw.get("tau_cell_overlap", 0.35)),
            min_stack_area_ratio=float(tau_raw.get("min_stack_area_ratio", 0.80)),
            min_complete_side_ratio=float(tau_raw.get("min_complete_side_ratio", 0.70)),
            max_same_level_overlap=float(tau_raw.get("max_same_level_overlap", 0.10)),
            overlap_warn_ratio=float(tau_raw.get("overlap_warn_ratio", 0.15)),
            partial_fit_tolerance=float(tau_raw.get("partial_fit_tolerance", 0.02)),
            max_duplicate_scale_ratio=float(tau_raw.get("max_duplicate_scale_ratio", 0.85)),
            free_gap_ratio=float(tau_raw.get("free_gap_ratio", 0.85)),
            min_support_coverage=min_support_coverage,
            max_support_ratio=max_support_ratio,
            max_bootstrap_combinations=max_bootstrap_combinations,
            max_position_correction=max_position_correction,
        )
    except KeyError as exc:
        raise ConfigError(f"falta la clave {exc} en {config_path}") from exc


def load_class_colors(config_path: Path) -> dict[str, tuple[int, int, int]]:
    """Carga colores RGB calibrados para las clases del renderer ISO."""
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        classes = raw.get("classes", {})
        return {
            str(name): tuple(int(channel) for channel in value["rgb"])
            for name, value in classes.items()
        }
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ConfigError(f"no se pudo leer la paleta de clases {config_path}") from exc


def load_isometric_config(config_path: Path) -> IsometricConfig:
    """Carga y valida configs/isometric.yaml una sola vez -- geometria
    del pallet y parametros visuales de la vista de inspeccion 3D.
    Se consume al servir /cam/<id>/iso."""
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"no se pudo leer {config_path}") from exc

    try:
        view_raw = raw["view"]

        return IsometricConfig(
            azimuth_deg=float(view_raw["azimuth_deg"]),
            elevation_deg=float(view_raw["elevation_deg"]),
            canvas_width=int(view_raw["canvas_width"]),
            canvas_height=int(view_raw["canvas_height"]),
            fill_margin=float(view_raw["fill_margin"]),
            visual_height_ratio=float(view_raw["visual_height_ratio"]),
            level_gap_ratio=float(view_raw["level_gap_ratio"]),
            class_colors=load_class_colors(config_path.with_name("color_cls.json")),
        )
    except KeyError as exc:
        raise ConfigError(f"falta la clave {exc} en {config_path}") from exc


def _color(raw: list) -> Color:
    b, g, r = raw
    return (int(b), int(g), int(r))


def load_drawing_config(config_path: Path) -> DrawingConfig:
    """Carga y valida configs/drawing.yaml una sola vez -- colores y
    layout de las anotaciones, separados de la config de pipeline."""
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"no se pudo leer {config_path}") from exc

    try:
        colors = raw["colors"]
        roi = raw["roi"]
        box = raw["box"]
        arm_alert = raw["arm_alert"]
        hud = raw["hud"]
        pallet = raw["pallet"]

        return DrawingConfig(
            color_new=_color(colors["new"]),
            color_redet=_color(colors["redet"]),
            color_pending=_color(colors["pending"]),
            color_roi=_color(colors["roi"]),
            color_text=_color(colors["text"]),
            color_hud_title=_color(colors["hud_title"]),
            color_arm_alert=_color(colors["arm_alert"]),
            level_colors=tuple(_color(color) for color in colors.get("levels", [])),
            roi_thickness=int(roi["thickness"]),
            box_thickness=int(box["thickness"]),
            box_label_font_scale=float(box["label_font_scale"]),
            box_label_thickness=int(box["label_thickness"]),
            box_circle_radius=int(box["circle_radius"]),
            arm_alert_font_scale=float(arm_alert["font_scale"]),
            arm_alert_thickness=int(arm_alert["thickness"]),
            hud_width=int(hud["width"]),
            hud_height=int(hud["height"]),
            hud_background=_color(hud["background"]),
            hud_title_font_scale=float(hud["title_font_scale"]),
            hud_title_thickness=int(hud["title_thickness"]),
            hud_line_font_scale=float(hud["line_font_scale"]),
            hud_line_thickness=int(hud["line_thickness"]),
            hud_visible_thickness=int(hud["visible_thickness"]),
            hud_fps_font_scale=float(hud["fps_font_scale"]),
            hud_fps_thickness=int(hud["fps_thickness"]),
            pallet_visible=bool(pallet.get("visible", True)),
            pallet_deck_thickness_m=float(pallet.get("deck_thickness_m", 0.08)),
            pallet_board_gap_m=float(pallet.get("board_gap_m", 0.012)),
            pallet_support_height_m=float(pallet.get("support_height_m", 0.07)),
            pallet_support_width_m=float(pallet.get("support_width_m", 0.065)),
            pallet_floor_margin_ratio=float(pallet.get("floor_margin_ratio", 0.25)),
        )
    except KeyError as exc:
        raise ConfigError(f"falta la clave {exc} en {config_path}") from exc
