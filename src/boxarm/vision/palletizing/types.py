from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm -- ver header completo en counter.py (paquete
# vision/palletizing/, historia acumulativa unica del modulo original).
# -----------------------------------------------------------------------
"""Tipos publicos del paquete: enums y dataclasses que describen el estado
y las decisiones de GridCounter, mas los alias de tipo de deteccion cruda.

No depende de ningun otro modulo del paquete."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Union

DetectionInput = Union[
    Tuple[int, int, int, int],
    Tuple[int, int, int, int, float],
    Tuple[int, int, int, int, float, str],
]
ParsedDetection = Tuple[Tuple[int, int, int, int], Optional[float], str]


class LevelSource(Enum):
    """Que mecanismo decidio el nivel de una deteccion.

    Existe porque hay varios y compiten: sin dejar constancia de cual gano,
    la explicacion de una caja mal ubicada solo vivia en el log y habia que
    reconstruirla a mano. Viaja en GridDetection para poder auditarla."""
    MATCH      = "emparejada"   # el matcher la asigno a una celda ya confirmada
    LADDER     = "escalera"     # escala aparente s(z)
    OCCLUSION  = "oclusion"     # par de detecciones solapadas del mismo frame
    STACKING   = "apilamiento"  # apoyo trabado sobre al menos dos celdas inferiores
    FLOOR      = "piso"         # no pisa nada: nivel 0
    GRAVITY    = "gravedad"     # bajado porque el nivel asignado no tenia soporte


@dataclass(frozen=True)
class LevelDecision:
    """Nivel resuelto mas quien lo resolvio."""
    level: int
    source: LevelSource
    reason: str = ""


class CellState(Enum):
    NEW      = "new"       # transicion 0 -> 1 en este frame: caja recien contada
    REDET    = "redet"     # celda ya ocupada, vuelta a observar (no se re-cuenta)
    VALIDATION = "validation"  # recorte consumido como evidencia; no crea ni modifica caja
    REJECTED = "rejected"  # no se pudo asignar con confianza (recorte, fuera de paleta, o F2)


@dataclass(frozen=True)
class GridDetection:
    """Resultado de asignar UNA deteccion a (celda, nivel) -- para dibujar
    y para decidir si actualiza chi. `cell`/`level` son None si state es
    REJECTED antes de poder asignarlos. `cell` es un indice en
    LevelLayout.cells del nivel asignado, no una coordenada fila/columna."""
    bbox: tuple[int, int, int, int]
    cell: int | None
    level: int | None
    state: CellState
    reason: str = ""  # motivo del rechazo, para logs (F2, recorte, fuera de rango, ...)
    confidence: float | None = None  # confianza YOLO, solo diagnostico/dibujo; no reemplaza la geometria
    level_source: "LevelSource | None" = None  # que mecanismo decidio `level` -- auditoria
    box_class: str = ""


class GateState(Enum):
    """Por que el contador esta o no validando cajas en este frame.

    Lo decide el cerebro, no el lazo de inferencia: es una regla de negocio
    (cuando es confiable mirar la paleta), no una cuestion de I/O."""
    COUNTING     = "validando"
    ARM_PAUSE    = "pausa-brazo"
    MOTION_PAUSE = "pausa-movimiento"
    SETTLING     = "esperando-estabilidad"


@dataclass(frozen=True)
class FrameInput:
    """Observacion CRUDA de un frame, tal como sale del detector.

    Nada aca viene decidido: `arm_visible` es lo que vio el modelo en este
    frame (sin debounce) y `motion_score` es el diff de grises sin
    umbralizar. Quien interpreta esos numeros es `GridCounter.update`."""
    boxes: list[DetectionInput]  # ya filtradas a la clase caja y a dentro del ROI
    arm_visible: bool
    motion_score: float


@dataclass(frozen=True)
class FrameResult:
    """Todo lo que el lazo de inferencia necesita para dibujar el frame."""
    detections: list[GridDetection]  # vacia si `gate` no es COUNTING
    gate: GateState
    gate_changed: bool   # `gate` difiere del frame anterior
    count_changed: bool  # el total cambio en este frame


@dataclass(frozen=True)
class SceneBox:
    """Una caja resuelta para dibujar.

    `side_a`/`side_b` son el consenso del nivel en el orden propio de esta
    caja (una caja girada 90 grados sigue viendose girada), y `z0`/`height`
    ya vienen apiladas: el renderizador no calcula nada. ``status`` separa
    geometria confirmada de observaciones visuales que todavia no pueden
    modificar el inventario."""
    cell: int
    level: int
    u: float
    v: float
    z0: float
    side_a: float
    side_b: float
    height: float
    box_class: str = ""
    status: str = "confirmed"


@dataclass(frozen=True)
class SceneOverlap:
    """Dos celdas confirmadas del mismo nivel cuyos footprints se pisan.

    Diagnostico: dos cajas de un nivel estan lado a lado, asi que un solape
    apreciable senala una celda duplicada (sobreconteo). El rectangulo ya
    viene calculado para que el ISO solo lo pinte."""
    cell_a: int
    cell_b: int
    level: int
    ratio: float
    u0: float
    v0: float
    u1: float
    v1: float
    z0: float
    height: float


@dataclass(frozen=True)
class SceneState:
    """Escena completa lista para renderizar. Es la unica salida que el ISO
    consume: si algo hay que decidir sobre la carga, se decide antes de
    construir esto."""
    boxes: list[SceneBox]
    overlaps: list[SceneOverlap]
    level_tops: list[float]  # cotas z acumuladas; len == levels + 1
    total_height: float
    total: int
    initial: int
    placed: int
    levels: int
    provisional_boxes: list[SceneBox] = field(default_factory=list)
    validating_initial: bool = False
