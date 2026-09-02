from __future__ import annotations

"""Compilacion unica de datos visuales a estructuras del runtime."""

from dataclasses import dataclass
from enum import Enum

from .box.coin_roll_100 import (
    COIN_ROLL_100_PATTERN_A,
    COIN_ROLL_100_PATTERN_B,
)
from .box.bag_10 import BAG_10_PATTERN_A, BAG_10_PATTERN_B
from .box.coin_roll_200 import (
    COIN_ROLL_200_PATTERN_A,
    COIN_ROLL_200_PATTERN_B,
)
from .box.coin_roll_10 import (
    COIN_ROLL_10_PATTERN_A,
    COIN_ROLL_10_PATTERN_B,
)
from .box.coin_roll_50 import (
    COIN_ROLL_50_PATTERN_A,
    COIN_ROLL_50_PATTERN_B,
)
from .dsl import HORIZONTAL, RawLayoutPattern


class BoxOrientation(Enum):
    HORIZONTAL = "H"
    VERTICAL = "V"


@dataclass(frozen=True)
class LayoutSlot:
    cell: int
    u: float
    v: float
    width: float
    height: float
    orientation: BoxOrientation


@dataclass(frozen=True)
class LayoutTemplate:
    box_class: str
    pattern: int  # 0=A, 1=B; NO es la paridad absoluta del nivel
    slots: tuple[LayoutSlot, ...]


ALL_BOX_CLASSES: tuple[str, ...] = (
    "bag_10",
    "bag_100",
    "bag_20",
    "bag_200",
    "bag_50",
    "bag_500",
    "coin_roll_10",
    "coin_roll_100",
    "coin_roll_20",
    "coin_roll_200",
    "coin_roll_50",
    "coin_roll_500",
)


def compile_pattern(
    box_class: str,
    pattern: int,
    raw: RawLayoutPattern,
) -> LayoutTemplate:
    """Compila una declaracion H/V sin alterar sus coordenadas ni su orden."""
    return LayoutTemplate(
        box_class=box_class,
        pattern=pattern,
        slots=tuple(
            LayoutSlot(
                cell=cell,
                u=u,
                v=v,
                width=width,
                height=height,
                orientation=(
                    BoxOrientation.HORIZONTAL
                    if orientation == HORIZONTAL
                    else BoxOrientation.VERTICAL
                ),
            )
            for cell, (u, v, width, height, orientation) in enumerate(raw)
        ),
    )


# Se compila exactamente una vez durante la importacion del modulo.
_TEMPLATES: dict[str, tuple[LayoutTemplate, LayoutTemplate]] = {
    "bag_10": (
        compile_pattern("bag_10", 0, BAG_10_PATTERN_A),
        compile_pattern("bag_10", 1, BAG_10_PATTERN_B),
    ),
    "coin_roll_10": (
        compile_pattern("coin_roll_10", 0, COIN_ROLL_10_PATTERN_A),
        compile_pattern("coin_roll_10", 1, COIN_ROLL_10_PATTERN_B),
    ),
    "coin_roll_50": (
        compile_pattern("coin_roll_50", 0, COIN_ROLL_50_PATTERN_A),
        compile_pattern("coin_roll_50", 1, COIN_ROLL_50_PATTERN_B),
    ),
    "coin_roll_100": (
        compile_pattern("coin_roll_100", 0, COIN_ROLL_100_PATTERN_A),
        compile_pattern("coin_roll_100", 1, COIN_ROLL_100_PATTERN_B),
    ),
    "coin_roll_200": (
        compile_pattern("coin_roll_200", 0, COIN_ROLL_200_PATTERN_A),
        compile_pattern("coin_roll_200", 1, COIN_ROLL_200_PATTERN_B),
    ),
}


def get_layout_template(
    box_class: str,
    level: int,
    phase: int = 0,
) -> LayoutTemplate | None:
    """Devuelve A/B segun nivel y fase elegida para la paleta actual."""
    templates = _TEMPLATES.get(box_class)
    if templates is None:
        return None
    return templates[(phase + level) % 2]


def get_layout_template_pair(
    box_class: str,
) -> tuple[LayoutTemplate, LayoutTemplate] | None:
    return _TEMPLATES.get(box_class)


def get_template_capacity(box_class: str) -> int | None:
    """Devuelve la capacidad por nivel desde el template de la clase."""
    templates = _TEMPLATES.get(box_class)
    if templates is None or not templates[0].slots or not templates[1].slots:
        return None
    capacity = len(templates[0].slots)
    if len(templates[1].slots) != capacity:
        raise ValueError(f"templates de {box_class!r} tienen capacidades distintas")
    return capacity


def get_template_box_classes() -> tuple[str, ...]:
    return tuple(sorted(_TEMPLATES))


def get_all_box_classes() -> tuple[str, ...]:
    """Devuelve las 12 clases previstas, tengan template o no."""
    return ALL_BOX_CLASSES


__all__ = [
    "BoxOrientation",
    "ALL_BOX_CLASSES",
    "LayoutSlot",
    "LayoutTemplate",
    "compile_pattern",
    "get_layout_template",
    "get_template_capacity",
    "get_all_box_classes",
]
