from __future__ import annotations

"""Fachada compatible para la API historica de plantillas.

Los datos por producto viven en :mod:`templates`; la compilacion, el
matching y el render estan separados en sus modulos especializados.
"""

# Al ejecutar esta fachada directamente, su carpeta contiene ``types.py`` y
# podria sombrear ``types`` de stdlib. Tambien agregamos ``src`` para que el
# comando funcione desde un checkout sin instalar el paquete.
import sys

_DIRECT_EXECUTION = __name__ == "__main__" and __package__ is None
if _DIRECT_EXECUTION and sys.path:
    # Debe ocurrir ANTES de importar pathlib/functools/dataclasses: todos ellos
    # terminan importando ``types`` de stdlib.
    sys.path.pop(0)

from pathlib import Path

if _DIRECT_EXECUTION:
    script_dir = Path(__file__).resolve().parent
    src_dir = str(script_dir.parents[2])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

from boxarm.vision.palletizing.templates.template_matcher import (
    TemplateAssignment,
    TemplateFit,
    TemplateObservation,
    fit_layout_hypothesis,
    match_layout_slot,
)
from boxarm.vision.palletizing.templates.template_render import (
    main as _main,
    render_layout_templates,
)
from boxarm.vision.palletizing.templates.template_runtime import (
    BoxOrientation,
    LayoutSlot,
    LayoutTemplate,
    get_layout_template,
)


__all__ = [
    "BoxOrientation",
    "LayoutSlot",
    "LayoutTemplate",
    "TemplateAssignment",
    "TemplateFit",
    "TemplateObservation",
    "fit_layout_hypothesis",
    "get_layout_template",
    "match_layout_slot",
    "render_layout_templates",
]


if __name__ == "__main__":
    raise SystemExit(_main())
