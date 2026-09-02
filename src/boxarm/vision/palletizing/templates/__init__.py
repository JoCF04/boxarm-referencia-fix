"""Datos estaticos de plantillas, organizados por clase de producto."""

from .box.coin_roll_100 import COIN_ROLL_100_PATTERN_A, COIN_ROLL_100_PATTERN_B
from .dsl import H, V, RawLayoutPattern, RawLayoutSlot, rotate_pattern, scale_pattern

__all__ = [
    "COIN_ROLL_100_PATTERN_A",
    "COIN_ROLL_100_PATTERN_B",
    "H",
    "RawLayoutPattern",
    "RawLayoutSlot",
    "rotate_pattern",
    "scale_pattern",
    "V",
]
