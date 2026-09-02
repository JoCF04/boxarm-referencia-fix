from __future__ import annotations

"""Datos visuales normalizados de paletizado para ``coin_roll_200``.

La referencia real se usa como Pattern A. Pattern B sigue la misma regla de
alternancia de coin_roll_100 y se obtiene escalando su Pattern A.
"""

from .coin_roll_100 import COIN_ROLL_100_PATTERN_A
from ..dsl import H, V, RawLayoutPattern, scale_pattern


# Pattern A: extraido con YOLO desde img_ref/coin_roll_200.jpg.
COIN_ROLL_200_PATTERN_A: RawLayoutPattern = (
    H(0.5554, 0.0761, 0.3259, 0.1457),
    H(0.2131, 0.0816, 0.3246, 0.1418),
    V(0.8244, 0.1380, 0.1628, 0.2950),
    H(0.3923, 0.2275, 0.3285, 0.1459),
    V(0.6613, 0.3005, 0.1603, 0.2984),
    V(0.1377, 0.3170, 0.1636, 0.2995),
    H(0.4033, 0.3786, 0.3300, 0.1486),
    V(0.8313, 0.4480, 0.1631, 0.3011),
    H(0.4082, 0.5361, 0.3326, 0.1489),
    V(0.6664, 0.6016, 0.1638, 0.2948),
    V(0.1446, 0.6155, 0.1669, 0.2932),
    H(0.4090, 0.6916, 0.3251, 0.1534),
    V(0.8374, 0.7700, 0.1646, 0.2955),
    H(0.5882, 0.8427, 0.3259, 0.1500),
    H(0.2210, 0.8445, 0.3241, 0.1541),
)


# Pattern B: segundo nivel conforme a la alternancia de coin_roll_100.
COIN_ROLL_200_PATTERN_B: RawLayoutPattern = scale_pattern(
    COIN_ROLL_100_PATTERN_A,
    1.05,
)


__all__ = ["COIN_ROLL_200_PATTERN_A", "COIN_ROLL_200_PATTERN_B"]
