from __future__ import annotations

"""Datos normalizados de templates para ``bag_10``.

Pattern A se extrajo con YOLO desde ``img_ref/bag_10.jpg``. Pattern B es la
misma distribucion rotada 180 grados.
"""

from ..dsl import H, RawLayoutPattern, V, rotate_pattern


BAG_10_PATTERN_A: RawLayoutPattern = (
    V(0.8320, 0.1835, 0.3184, 0.3672),
    V(0.4970, 0.1817, 0.3167, 0.3634),
    V(0.1830, 0.1808, 0.3093, 0.3617),
    H(0.7430, 0.5250, 0.3861, 0.4200),
    H(0.2510, 0.5241, 0.3762, 0.4262),
    H(0.2500, 0.8350, 0.3725, 0.4323),
    H(0.7450, 0.8353, 0.3830, 0.4317),
)
BAG_10_PATTERN_B: RawLayoutPattern = rotate_pattern(BAG_10_PATTERN_A, 180)


__all__ = ["BAG_10_PATTERN_A", "BAG_10_PATTERN_B"]
