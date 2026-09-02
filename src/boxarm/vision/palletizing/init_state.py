from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm -- ver header completo en counter.py (paquete
# vision/palletizing/, historia acumulativa unica del modulo original).
# -----------------------------------------------------------------------
"""Mixin _InitStateMixin: construccion de GridCounter, persistencia de
estado (state_dict/save_state/load_state) y geometria basica compartida
(proyeccion, footprint medido, posicion de celda)."""

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from boxarm.config import PalletizingConfig

from .formulas import (
    _build_scale_ladder,
    _footprint_overlap_over_min,
    _ladder_step_gap,
    _measure_footprint as _measure_footprint_formula,
    _observed_median,
    _project,
    build_homography,
)
from .types import GateState
from .templates.template_runtime import get_template_capacity

logger = logging.getLogger(__name__)


class _CameraLoggerAdapter(logging.LoggerAdapter):
    """Antepone "[tag] " a cada mensaje: con 3 GridCounter (uno por camara)
    escribiendo al mismo logger en paralelo, sin esto "caja contada" no dice
    de cual proceso vino y es imposible confirmar a ojo si una camara
    puntual dejo de contar."""

    def process(self, msg, kwargs):
        return self.extra["prefix"] + msg, kwargs


class _InitStateMixin:
    """Construccion, estado persistente y geometria basica de GridCounter."""

    def __init__(self, pallet_pts: np.ndarray, cfg: PalletizingConfig, cam_tag: str = "") -> None:
        """`pallet_pts` son las 4 esquinas de la PALETA en pixeles del frame
        (pallet_roi de configs/roi_cam_<id>.json), no el ROI de deteccion:
        el cuadrado unidad de todo el conteo es la superficie de la tarima.

        `cam_tag` (p.ej. "Camara 2") prefija todo lo que loguea este
        contador -- con 3 procesos corriendo en paralelo, "caja contada"
        sin decir de que camara viene hacia imposible confirmar a ojo si
        una camara puntual dejo de contar."""
        self._cfg = cfg
        prefix = f"[{cam_tag}] " if cam_tag else ""
        self._log = _CameraLoggerAdapter(logger, {"prefix": prefix})
        self._homography = build_homography(pallet_pts)
        # Inversa: una geometria corregida en el cuadrado unidad (por ejemplo
        # un recorte completado) tiene que volver a pixeles para reentrar por
        # los mismos caminos que una deteccion cruda.
        self._homography_inv = np.linalg.inv(self._homography)
        # s(z) = reference_scale_px * (c_z - box_height) / (c_z - (z+1)*box_height)
        # -- formula de la seccion 5.A, reparametrizada para calibrarse con
        # una unica medicion directa (reference_scale_px) en vez de fL y C_z
        # por separado (ver docs/palletizing_counting.md seccion 6.C).
        self._ladder = _build_scale_ladder(
            cfg.reference_scale_px, cfg.c_z, cfg.box_height, cfg.levels,
        )
        self._check_ladder_is_separable()
        # -- Gate: cuando es confiable mirar la paleta ----------------------
        # Vivia en el lazo de inferencia, que es I/O y no deberia decidir
        # nada. Son reglas de negocio: mientras el brazo esta en escena o
        # algo se mueve, lo que se ve no es la carga en reposo. NO se reinicia
        # en reset_pallet(): describe el movimiento/brazo de ESTE frame, no a
        # que paleta pertenece lo que se cuenta.
        self._gate = GateState.SETTLING
        self._stable_frames = 0
        # Frames consecutivos SIN brazo. El ciclo se cierra recien cuando
        # supera `gate.arm_debounce_frames`: antes bastaba un solo frame sin
        # brazo para cerrarlo, asi que un unico fallo de deteccion en medio
        # de la maniobra abria la puerta a contar otra caja.
        self._frames_without_arm = 0
        self._arm_seen_in_cycle = False
        self.reset_pallet()

    def reset_pallet(self) -> None:
        """Vuelve a cero TODO el estado de la carga actual -- misma
        inicializacion que `__init__` corria para la primera paleta,
        factorizada aca para poder repetirla cuando la paleta fisica se
        vacia a mitad de una corrida.

        Dispara esto `_FrameLoopMixin._count_boxes()` cuando pasan
        `gate.empty_pallet_debounce_frames` frames COUNTING seguidos sin
        ninguna deteccion habiendo cajas confirmadas (ver frame_loop.py) --
        es la unica transicion 1->0 de `chi(g,z)` de todo el paquete, y por
        eso vive en un metodo con nombre propio en vez de mezclarse con la
        transicion 0->1 de `_count_boxes`.

        NO toca calibracion/geometria de camara (`_cfg`, `_homography*`,
        `_ladder`) ni el estado del gate de movimiento/brazo del frame
        actual (`_gate`, `_stable_frames`, `_frames_without_arm`,
        `_arm_seen_in_cycle`): esas dos cosas describen la CAMARA y EL
        INSTANTE, no la carga -- siguen siendo la misma tarima fisica y el
        mismo frame, solo cambio que hay apoyado en ella."""
        self._occupied: set[tuple[int, int]] = set()  # chi(g,z) -- solo crece dentro de una paleta
        # En layout_mode=auto cada nivel descubre sus posiciones desde
        # centroides proyectados, en vez de imponer una rejilla ficticia.
        self._dynamic_positions: dict[tuple[int, int], tuple[float, float]] = {}
        self._next_cell_by_level: dict[int, int] = {}
        # Footprint (du, dv) MEDIDO en [0,1]^2 de cada celda confirmada,
        # tomado de su bbox real via la homografia -- no un tamano fijo
        # de config. Se guarda una sola vez, al confirmar (NEW), igual
        # Lo consume isometric.py.
        self._footprint: dict[tuple[int, int], tuple[float, float]] = {}
        # Tamano canonico de dibujo por nivel. Se fija con la primera caja
        # confirmada y despues NO cambia: las detecciones posteriores son
        # observaciones, no permiso para deformar el estado persistente.
        self._level_footprint: dict[int, tuple[float, float]] = {}
        # Frame (llamada a update()) en que se confirmo cada celda. Sirve
        # para distinguir apilamiento real de deteccion duplicada: ver
        # _stacking_level.
        self._current_frame = 0
        self._cell_frame: dict[tuple[int, int], int] = {}
        # El brazo apoya UNA caja por ciclo (entra, deja, sale). Cada ciclo
        # cerrado concede un credito de colocacion. Si vision omite la caja,
        # el credito queda pendiente para recuperar el backlog mas adelante.
        # Mientras no haya pasado ningun ciclo se acepta cualquier cantidad:
        # es el inventario inicial de una paleta que ya venia cargada.
        self._arm_cycle_seen = False
        self._placement_credits = 0
        # Niveles probados llenos por evidencia de oclusion durante el
        # inventario inicial: ver el encadenamiento en `_count_boxes`. No es
        # lo mismo que "tiene n celdas confirmadas" -- justamente existe para
        # los niveles cuya cuenta exacta es inalcanzable porque parte de sus
        # cajas nacieron tapadas.
        self._proven_full: set[int] = set()
        # Ventana LOCAL del seguimiento. Los niveles almacenados conservan su
        # indice absoluto para persistencia/ISO; solo matching y decision de
        # nuevas cajas ignoran los niveles absolutos menores que este piso.
        self._tracking_floor_level = 0
        # La paleta puede arrancar con varios niveles ya construidos. Antes
        # del tracking normal se permite UNA reconciliacion conjunta de las
        # observaciones iniciales; una vez encontrada una solucion unica, no
        # se vuelven a mover identidades entre niveles.
        self._bootstrap_reconciled = False
        # Una clase con template puede arrancar a mitad de operacion, incluso
        # con el brazo ya dentro. Hasta aceptar el primer bloque estable, las
        # cajas visibles forman la linea base inicial y no consumen creditos.
        self._template_baseline_pending = False
        # Fase del patron para ESTA paleta: 0 => N0=A,N1=B; 1 => N0=B,N1=A.
        # Se descubre una sola vez durante bootstrap y luego solo alterna.
        self._template_phase: int | None = None
        # Registro del template dentro del ROI amplio: sx, sy, tx, ty.
        # La clase define topologia; cada camara aporta posicion/escala.
        self._template_registration: tuple[float, float, float, float] | None = None
        self._last_bootstrap_signature: tuple | None = None
        self._bootstrap_signature_hits = 0
        # Cache de _select_template_bootstrap_fit() por entrada exacta: esa
        # funcion termina llamando a _estimate_registration(), que compara
        # cada observacion contra cada slot DOS veces anidado (score() dentro
        # del bucle de candidatos) -- con una clase de capacidad grande
        # (coin_roll_10=25) son cientos de miles de operaciones en Python
        # puro, TODAS desde cero, en CADA frame mientras el bootstrap no se
        # resuelve. La escena real no cambia frame a frame mientras el video
        # esta quieto (o casi): si las mismas cajas/fragmentos ya se vieron
        # en el frame anterior, se reusa el resultado en vez de recalcular.
        # Ver faulthandler dump que encontro este cuello de botella real
        # (BOXARM_STALL_DUMP=1): template_matcher.py:158 score(),
        # llamado desde _estimate_registration() en bootstrap.py:297.
        self._template_fit_cache_key: tuple | None = None
        self._template_fit_cache_value: object = None
        self._bootstrap_solution_history: list[dict] = []
        self._last_bootstrap_partials: list[tuple[float, float, float, float]] = []
        self._last_bootstrap_complete_count = 0
        # Frontera consecutiva que se esta resolviendo: i -> i+1. El mismo
        # solver se reutiliza desde el piso hasta el ultimo nivel observado.
        self._bootstrap_level = 0
        self._bootstrap_consumed_partials: list[tuple[float, float, float, float]] = []
        # El inventario inicial puede estar temporalmente mal repartido entre
        # Ni/N(i+1) mientras el solver valida su unica clase fisica. Ese estado es
        # evidencia interna, NO una escena publicable: el ISO permanece vacio
        # hasta que la correccion alcance `confirmation.min_stable`.
        self._initial_scene_deferred = False
        # Celdas ya rechazadas en el ciclo actual -- solo para no repetir el
        # mismo warning en cada frame mientras la caja siga a la vista.
        self._rejected_in_cycle: set[tuple[int, int]] = set()
        # Diagnosticos geometricos ya emitidos durante el ciclo. Una caja
        # estable no debe imprimir el mismo analisis varias veces por segundo.
        self._logged_candidate_diagnostics: set[tuple[float, float, float, float]] = set()
        # Candidatas aun no persistidas: historial corto de bboxes en frames
        # COUNTING consecutivos. Es una confirmacion temporal cuya mediana,
        # NO una identidad de tracking; al confirmar se descarta y manda
        # exclusivamente (celda, nivel).
        self._pending_candidates: list[list[tuple[int, int, int, int]]] = []
        # total = cajas en la paleta (lo que hay). Se desglosa en dos, que
        # NO son lo mismo y antes se mostraban mezcladas bajo "Colocadas":
        #   initial  -> ya estaban cuando arranco el video (paleta a medio
        #               cargar, que es el caso real: este video empieza con
        #               14 puestas y no existe grabacion desde cero)
        #   placed   -> las que el brazo apoyo mientras mirabamos
        self.total = 0
        self.initial = 0
        self.placed = 0
        self.visible = 0
        # Clase de caja que se esta paletizando (caja_1sol, caja_2soles, ...).
        # La capacidad de un nivel depende de ella: la paleta es la misma pero
        # la caja de cada denominacion mide distinto. La fija inferencia al
        # resolver la clase del modelo; hasta entonces no hay capacidad
        # declarada y _level_is_full cae en la deteccion de huecos.
        self._box_class: str | None = None
        # Ultima composicion por nivel ya reportada (ver _log_level_composition).
        self._last_level_signature: tuple | None = None
        # Frames COUNTING consecutivos sin ninguna deteccion -- ver
        # gate.empty_pallet_debounce_frames y _count_boxes().
        self._empty_frames = 0

    @property
    def cfg(self) -> PalletizingConfig:
        return self._cfg

    @property
    def arm_cycle_seen(self) -> bool:
        """True desde que el brazo completo su primer viaje.

        Antes de eso, lo que hay en la paleta es inventario inicial y se
        acepta cualquier cantidad de cajas nuevas; despues, cada caja nueva
        necesita consumir un credito concedido por un ciclo cerrado."""
        return self._arm_cycle_seen

    def set_box_class(self, box_class: str) -> None:
        """Declara que denominacion se esta paletizando, para elegir la
        capacidad por nivel desde el template de la clase. La llama inferencia una
        vez, con el nombre de clase que reporta el modelo."""
        if box_class == self._box_class:
            return
        self._box_class = box_class
        self._template_phase = None
        self._template_registration = None
        capacity = get_template_capacity(box_class)
        if capacity is None:
            self._log.warning(
                "clase %r sin template registrado -- el nivel lleno se va a "
                "deducir buscando huecos en la ROI, que solo funciona si la ROI esta ajustada "
                "a la paleta real", box_class,
            )
        else:
            if not self._bootstrap_reconciled and self.total == 0:
                self._template_baseline_pending = True
            self._log.info(
                "clase activa=%r -- template cargado, capacidad=%d cajas por nivel",
                box_class,
                capacity,
            )

    def state_dict(self, height_ratio: float) -> dict:
        """Estado canónico versionado, suficiente para restaurar la paleta."""
        levels = []
        for level in range(self._cfg.levels):
            boxes = []
            for cell, z in sorted(self._occupied):
                if z != level or (cell, z) not in self._footprint:
                    continue
                u, v = self._cell_position(cell, z)
                side_a, side_b = self._footprint[(cell, z)]
                boxes.append({
                    "cell": cell,
                    "u": u,
                    "v": v,
                    "side_a": side_a,
                    "side_b": side_b,
                    "confirmed_frame": self._cell_frame.get((cell, z), 0),
                })
            if boxes:
                consensus = self._level_footprint.get(level)
                levels.append({
                    "level": level,
                    "long_median": None if consensus is None else consensus[0],
                    "short_median": None if consensus is None else consensus[1],
                    "boxes": boxes,
                })
        return {
            "schema_version": 1,
            "active_box_class": self._box_class,
            "template_phase": self._template_phase,
            "template_registration": self._template_registration,
            "capacity_per_level": get_template_capacity(self._box_class or ""),
            "counts": {
                "total": self.total,
                "initial": self.initial,
                "placed": self.placed,
            },
            "arm_cycle_seen": self._arm_cycle_seen,
            "placement_credits": self._placement_credits,
            "levels": levels,
            "scene": asdict(self.scene_state(height_ratio)),
        }

    def save_state(self, path: Path, height_ratio: float) -> None:
        """Persiste JSON con reemplazo atómico; nunca deja un archivo parcial."""
        if self._initial_scene_deferred and not self._bootstrap_reconciled:
            # No convertir una hipotesis provisional en estado restaurable.
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        encoded = json.dumps(
            self.state_dict(height_ratio), ensure_ascii=False, indent=2,
        ) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def load_state(self, path: Path) -> None:
        """Valida completamente un JSON antes de mutar el contador."""
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"estado de paleta ilegible: {path}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("schema_version de estado de paleta incompatible")
        if self._box_class is None:
            raise ValueError("set_box_class debe ejecutarse antes de restaurar estado")
        if payload.get("active_box_class") != self._box_class:
            raise ValueError(
                f"estado pertenece a {payload.get('active_box_class')!r}, no a {self._box_class!r}"
            )

        counts = payload.get("counts")
        levels = payload.get("levels")
        if not isinstance(counts, dict) or not isinstance(levels, list):
            raise ValueError("estado sin counts/levels válidos")

        occupied: set[tuple[int, int]] = set()
        positions: dict[tuple[int, int], tuple[float, float]] = {}
        footprints: dict[tuple[int, int], tuple[float, float]] = {}
        cell_frames: dict[tuple[int, int], int] = {}
        per_level: dict[int, int] = {}
        for level_record in levels:
            if not isinstance(level_record, dict):
                raise ValueError("registro de nivel inválido")
            level = int(level_record.get("level", -1))
            if not 0 <= level < self._cfg.levels:
                raise ValueError(f"nivel fuera de rango: {level}")
            boxes = level_record.get("boxes")
            if not isinstance(boxes, list):
                raise ValueError(f"boxes inválido en nivel {level}")
            for box in boxes:
                try:
                    cell = int(box["cell"])
                    u, v = float(box["u"]), float(box["v"])
                    side_a, side_b = float(box["side_a"]), float(box["side_b"])
                    confirmed_frame = int(box.get("confirmed_frame", 0))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"caja inválida en nivel {level}") from exc
                key = (cell, level)
                if key in occupied:
                    raise ValueError(f"identidad duplicada: {key}")
                if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                    raise ValueError(f"posición fuera de paleta: {key}")
                if side_a <= 0.0 or side_b <= 0.0:
                    raise ValueError(f"footprint no positivo: {key}")
                occupied.add(key)
                positions[key] = (u, v)
                footprints[key] = (side_a, side_b)
                cell_frames[key] = confirmed_frame
                per_level[level] = per_level.get(level, 0) + 1

        total = int(counts.get("total", -1))
        initial = int(counts.get("initial", -1))
        placed = int(counts.get("placed", -1))
        if total != len(occupied) or initial < 0 or placed < 0 or initial + placed != total:
            raise ValueError("conteos incompatibles con las cajas persistidas")
        placement_credits = payload.get("placement_credits", 0)
        if (
            isinstance(placement_credits, bool)
            or not isinstance(placement_credits, int)
            or placement_credits < 0
        ):
            raise ValueError("placement_credits invalido")
        capacity = get_template_capacity(self._box_class or "")
        if capacity is not None and any(count > capacity for count in per_level.values()):
            raise ValueError("un nivel persistido excede su capacidad exacta")
        template_phase = payload.get("template_phase")
        if template_phase is not None and template_phase not in (0, 1):
            raise ValueError("template_phase debe ser null, 0 (A) o 1 (B)")
        template_registration = payload.get("template_registration")
        if template_registration is not None:
            if (
                not isinstance(template_registration, list)
                or len(template_registration) != 4
            ):
                raise ValueError("template_registration debe contener sx, sy, tx, ty")
            template_registration = tuple(float(value) for value in template_registration)
            if template_registration[0] <= 0.0 or template_registration[1] <= 0.0:
                raise ValueError("template_registration requiere escalas positivas")

        level_footprints: dict[int, tuple[float, float]] = {}
        for level in per_level:
            own = [fp for (_g, z), fp in footprints.items() if z == level]
            level_footprints[level] = (
                _observed_median([max(fp) for fp in own]),
                _observed_median([min(fp) for fp in own]),
            )
        overlap_limit = getattr(self._cfg, "max_same_level_overlap", 0.10)
        keys = sorted(occupied)
        for index, key_a in enumerate(keys):
            for key_b in keys[index + 1:]:
                if key_a[1] != key_b[1]:
                    continue
                long_side, short_side = level_footprints[key_a[1]]
                fp_a = (long_side, short_side) if footprints[key_a][0] >= footprints[key_a][1] else (short_side, long_side)
                fp_b = (long_side, short_side) if footprints[key_b][0] >= footprints[key_b][1] else (short_side, long_side)
                if _footprint_overlap_over_min(
                    positions[key_a], fp_a, positions[key_b], fp_b,
                ) > overlap_limit:
                    raise ValueError(f"interpenetración intranivel persistida: {key_a} y {key_b}")

        # Commit único: ninguna validación posterior puede dejar estado parcial.
        self._occupied = occupied
        self._dynamic_positions = positions if self._cfg.layout_mode == "auto" else {}
        self._footprint = footprints
        self._level_footprint = level_footprints
        self._cell_frame = cell_frames
        self._next_cell_by_level = {
            level: max(cell for cell, z in occupied if z == level) + 1
            for level in per_level
        }
        self._current_frame = max(cell_frames.values(), default=0) + 1
        self.total, self.initial, self.placed = total, initial, placed
        self.visible = 0
        self._arm_cycle_seen = bool(payload.get("arm_cycle_seen", total > 0))
        self._placement_credits = placement_credits
        self._template_phase = (
            None if template_phase is None else int(template_phase)
        )
        self._template_registration = template_registration
        # Que exista una caja en el nivel z implica que z-1 estaba lleno
        # cuando se confirmo: se deduce del propio estado en vez de agregar
        # un campo al esquema.
        self._proven_full = {z - 1 for (_g, z) in occupied if z > 0}
        capacity = get_template_capacity(self._box_class or "")
        full_levels = [
            level for level, count in per_level.items()
            if capacity is not None and count == capacity
        ]
        self._tracking_floor_level = max(full_levels, default=0)
        # Un snapshot validado ya es una composicion canonica; volver a
        # ejecutar el solver inicial podria reasignar identidades restauradas.
        self._bootstrap_reconciled = True
        self._template_baseline_pending = False
        self._last_bootstrap_signature = None
        self._bootstrap_signature_hits = 0
        self._bootstrap_solution_history.clear()
        self._last_bootstrap_partials.clear()
        self._last_bootstrap_complete_count = 0
        self._bootstrap_level = 0
        self._bootstrap_consumed_partials.clear()
        self._initial_scene_deferred = False
        self._pending_candidates.clear()
        self._rejected_in_cycle.clear()
        self._logged_candidate_diagnostics.clear()

    def _check_ladder_is_separable(self) -> None:
        """Avisa si la escalera s(z) y tau_rung hacen los niveles
        indistinguibles POR CONSTRUCCION.

        `_assign_level` acepta un peldano cuando el error relativo es
        <= tau_rung, asi que cada peldano cubre una banda de +-tau_rung.
        Para que dos peldanos consecutivos no se solapen hace falta

            tau_rung < (s(z+1) - s(z)) / (2 * s(z))

        Si no se cumple, la banda del nivel bajo se traga el peldano del
        alto: el ruido del bbox decide el nivel y, como min() desempata
        hacia el indice smaller, todo tiende a colapsar al nivel 0 -- que se
        ve como una pila entera pintada de un solo color en la vista iso.
        """
        for z in range(len(self._ladder) - 1):
            s_low, s_high = self._ladder[z], self._ladder[z + 1]
            half_gap = _ladder_step_gap(s_low, s_high)
            if self._cfg.tau_rung >= half_gap:
                self._log.warning(
                    "calibracion: la escalera no separa los niveles %d y %d (s=%.1f y %.1f px, "
                    "%.1f%% de diferencia contra tau_rung=%.1f%%, haria falta < %.1f%%). El nivel "
                    "se decide por apilamiento geometrico, asi que esto solo afecta a una caja "
                    "que no pise a ninguna otra. Se corrige con c_z/box_height en palletizing.yaml.",
                    z, z + 1, s_low, s_high,
                    (s_high / s_low - 1.0) * 100.0,
                    self._cfg.tau_rung * 100.0,
                    half_gap * 100.0,
                )

    def _unproject(self, u: float, v: float) -> tuple[float, float]:
        """Del cuadrado unidad de vuelta a pixeles de la imagen."""
        return _project(self._homography_inv, u, v)

    def _cell_position(self, cell: int, level: int) -> tuple[float, float]:
        """Centro normalizado de una caja, descubierto o configurado."""
        if self._cfg.layout_mode == "auto":
            return self._dynamic_positions[(cell, level)]
        return self._cfg.levels_layout[level].cells[cell]

    def _measure_footprint(self, cx: float, cy: float, w_px: float, h_px: float) -> tuple[float, float]:
        """Ancho/alto REAL de esta deteccion en [0,1]^2, medido proyectando
        los bordes de su propio bbox por la homografia -- no un tamano de
        config. Sirve de base a isometric.py para el footprint 3D."""
        return _measure_footprint_formula(self._homography, cx, cy, w_px, h_px)
