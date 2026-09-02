from __future__ import annotations

"""Datos visuales normalizados de paletizado para ``coin_roll_10``.

La plantilla base se extrajo con el modelo desde ``img_ref/coin_roll_10.jpg``.
El segundo patron es la misma topologia rotada 180 grados.
"""

from ..dsl import H, V, RawLayoutPattern, rotate_pattern


# Pattern A: img_ref/coin_roll_10.jpg (25 cajas por nivel).
COIN_ROLL_10_PATTERN_A: RawLayoutPattern = (
    H(0.6785, 0.0809, 0.2487, 0.1286),
    H(0.4146, 0.0828, 0.2456, 0.1280),
    H(0.1641, 0.0865, 0.2410, 0.1318),
    V(0.8941, 0.1047, 0.1503, 0.2075),
    H(0.6782, 0.2116, 0.2505, 0.1273),
    H(0.4113, 0.2138, 0.2438, 0.1256),
    H(0.1600, 0.2204, 0.2438, 0.1312),
    H(0.6779, 0.3447, 0.2446, 0.1284),
    H(0.4228, 0.3535, 0.2454, 0.1301),
    H(0.1656, 0.3585, 0.2449, 0.1273),
    V(0.8967, 0.3705, 0.1503, 0.2101),
    H(0.6872, 0.4858, 0.2487, 0.1243),
    H(0.4233, 0.4910, 0.2451, 0.1275),
    H(0.1679, 0.4938, 0.2456, 0.1286),
    H(0.6887, 0.6211, 0.2438, 0.1286),
    H(0.4290, 0.6280, 0.2456, 0.1295),
    H(0.1718, 0.6303, 0.2459, 0.1277),
    V(0.9028, 0.6333, 0.1492, 0.2099),
    H(0.6941, 0.7697, 0.2451, 0.1308),
    H(0.1769, 0.7729, 0.2436, 0.1265),
    H(0.4303, 0.7731, 0.2446, 0.1271),
    V(0.9062, 0.8626, 0.1515, 0.2080),
    H(0.6933, 0.9006, 0.2449, 0.1277),
    H(0.4362, 0.9030, 0.2449, 0.1267),
    H(0.1800, 0.9045, 0.2436, 0.1277),
)


# Pattern B: misma distribucion fisica, rotada 180 grados.
COIN_ROLL_10_PATTERN_B: RawLayoutPattern = rotate_pattern(COIN_ROLL_10_PATTERN_A, 180)


__all__ = ["COIN_ROLL_10_PATTERN_A", "COIN_ROLL_10_PATTERN_B"]
