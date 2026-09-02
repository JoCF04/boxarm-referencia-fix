from __future__ import annotations

"""DSL minimo para declarar posiciones fisicas de una plantilla.

Los helpers solo construyen tuplas inmutables. No contienen matching,
conversiones de coordenadas ni objetos del runtime.
"""

from typing import Final, Tuple


HORIZONTAL: Final = 0
VERTICAL: Final = 1

RawLayoutSlot = Tuple[float, float, float, float, int]
RawLayoutPattern = Tuple[RawLayoutSlot, ...]


def H(cx: float, cy: float, width: float, height: float) -> RawLayoutSlot:
    return (cx, cy, width, height, HORIZONTAL)


def V(cx: float, cy: float, width: float, height: float) -> RawLayoutSlot:
    return (cx, cy, width, height, VERTICAL)


def rotate_pattern(pattern: RawLayoutPattern, degrees: int) -> RawLayoutPattern:
    """Rota centros normalizados y reutiliza una plantilla en otro angulo.

    ``degrees`` admite 0, 90, 180 y 270 en sentido antihorario. En 90/270
    también se intercambian dimensiones y orientación; en 180 se conservan.
    """
    angle = degrees % 360
    if angle not in (0, 90, 180, 270):
        raise ValueError("degrees must be a multiple of 90")
    if angle == 0:
        return pattern

    rotated: list[RawLayoutSlot] = []
    for u, v, width, height, orientation in pattern:
        if angle == 90:
            new_u, new_v = v, 1.0 - u
        elif angle == 180:
            new_u, new_v = 1.0 - u, 1.0 - v
        else:  # 270
            new_u, new_v = 1.0 - v, u

        if angle in (90, 270):
            new_width, new_height = height, width
            new_orientation = VERTICAL if orientation == HORIZONTAL else HORIZONTAL
        else:
            new_width, new_height = width, height
            new_orientation = orientation

        rotated.append((new_u, new_v, new_width, new_height, new_orientation))
    return tuple(rotated)


def scale_pattern(
    pattern: RawLayoutPattern,
    factor: float,
    center: tuple[float, float] = (0.5, 0.5),
) -> RawLayoutPattern:
    """Escala una plantilla normalizada alrededor de un centro fijo."""
    if factor <= 0:
        raise ValueError("factor must be greater than zero")
    cx, cy = center
    return tuple(
        (
            cx + (u - cx) * factor,
            cy + (v - cy) * factor,
            width * factor,
            height * factor,
            orientation,
        )
        for u, v, width, height, orientation in pattern
    )


__all__ = [
    "H",
    "HORIZONTAL",
    "RawLayoutPattern",
    "RawLayoutSlot",
    "rotate_pattern",
    "scale_pattern",
    "V",
    "VERTICAL",
]
