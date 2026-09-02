from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm -- ver header completo en counter.py (paquete
# vision/palletizing/, historia acumulativa unica del modulo original).
# -----------------------------------------------------------------------
"""Reexporta la interfaz publica del paquete de paletizado (P-7): nada de
logica vive aca."""

from .counter import GridCounter
from .formulas import _observed_median, build_homography
from .layout_templates import (
    BoxOrientation,
    LayoutSlot,
    LayoutTemplate,
    TemplateAssignment,
    TemplateFit,
    TemplateObservation,
    fit_layout_hypothesis,
    get_layout_template,
    match_layout_slot,
    render_layout_templates,
)
from .types import (
    CellState,
    DetectionInput,
    FrameInput,
    FrameResult,
    GateState,
    GridDetection,
    LevelDecision,
    LevelSource,
    ParsedDetection,
    SceneBox,
    SceneOverlap,
    SceneState,
)

__all__ = [
    "GridCounter",
    "build_homography",
    "BoxOrientation",
    "CellState",
    "DetectionInput",
    "FrameInput",
    "FrameResult",
    "GateState",
    "GridDetection",
    "LevelDecision",
    "LevelSource",
    "LayoutSlot",
    "LayoutTemplate",
    "TemplateAssignment",
    "TemplateFit",
    "TemplateObservation",
    "ParsedDetection",
    "SceneBox",
    "SceneOverlap",
    "SceneState",
    "fit_layout_hypothesis",
    "get_layout_template",
    "match_layout_slot",
    "render_layout_templates",
    "_observed_median",
]
