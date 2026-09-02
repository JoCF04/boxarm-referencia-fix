from __future__ import annotations

"""Datos visuales normalizados de paletizado para ``coin_roll_50``.

Las dos plantillas se extrajeron con el modelo de cajas desde:
``videos/img_ref/coin_roll_50.jpg`` y ``coin_roll_50_1.jpg``.
"""

from ..dsl import H, V, RawLayoutPattern, rotate_pattern


# Pattern A: coin_roll_50.jpg
COIN_ROLL_50_PATTERN_A: RawLayoutPattern = (
    H(0.6410, 0.1192, 0.2766, 0.1572),
    H(0.3410, 0.1253, 0.2771, 0.1525),
    V(0.8777, 0.1867, 0.1480, 0.2942),
    V(0.1199, 0.1948, 0.1442, 0.2904),
    H(0.5042, 0.2735, 0.2752, 0.1560),
    V(0.7297, 0.3493, 0.1501, 0.2916),
    V(0.2712, 0.3566, 0.1469, 0.2850),
    V(0.5754, 0.4941, 0.1477, 0.2940),
    V(0.4222, 0.4971, 0.1475, 0.2870),
    V(0.8849, 0.4999, 0.1446, 0.2925),
    V(0.1292, 0.5140, 0.1387, 0.2867),
    V(0.7362, 0.6485, 0.1453, 0.2885),
    V(0.2754, 0.6563, 0.1440, 0.2855),
    H(0.4970, 0.7348, 0.2759, 0.1591),
    V(0.8885, 0.8108, 0.1424, 0.2884),
    V(0.1359, 0.8322, 0.1380, 0.2847),
    H(0.6589, 0.8912, 0.2788, 0.1556),
    H(0.3607, 0.8954, 0.2744, 0.1558),
)


# Pattern B extraido: coin_roll_50_1.jpg.
# Se conserva como referencia de auditoria. La plantilla activa B se genera
# abajo rotando A, porque la segunda imagen es la misma topologia girada.
COIN_ROLL_50_PATTERN_B_EXTRACTED: RawLayoutPattern = (
    H(0.8226, 0.0993, 0.2786, 0.1623),
    H(0.4971, 0.1090, 0.2861, 0.1564),
    H(0.1852, 0.1165, 0.2771, 0.1557),
    H(0.6530, 0.2623, 0.2810, 0.1616),
    H(0.3417, 0.2682, 0.2820, 0.1587),
    V(0.8928, 0.3403, 0.1499, 0.3056),
    V(0.1096, 0.3487, 0.1449, 0.2961),
    H(0.5017, 0.4217, 0.2793, 0.1609),
    V(0.7362, 0.4981, 0.1509, 0.3022),
    V(0.2632, 0.5021, 0.1496, 0.2994),
    H(0.5025, 0.5788, 0.2882, 0.1661),
    V(0.9010, 0.6531, 0.1493, 0.3000),
    V(0.1133, 0.6665, 0.1470, 0.2960),
    H(0.6562, 0.7437, 0.2878, 0.1647),
    H(0.3468, 0.7496, 0.2858, 0.1620),
    H(0.8301, 0.8978, 0.2795, 0.1566),
    H(0.5042, 0.9048, 0.2853, 0.1570),
    H(0.2005, 0.9098, 0.2775, 0.1562),
)


# Pattern B activo: misma plantilla fisica que A, rotada 90 grados.
COIN_ROLL_50_PATTERN_B: RawLayoutPattern = rotate_pattern(COIN_ROLL_50_PATTERN_A, 90)


__all__ = [
    "COIN_ROLL_50_PATTERN_A",
    "COIN_ROLL_50_PATTERN_B",
    "COIN_ROLL_50_PATTERN_B_EXTRACTED",
]
