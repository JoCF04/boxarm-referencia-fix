from __future__ import annotations

"""Matching geometrico de observaciones contra plantillas compiladas."""

from dataclasses import dataclass
from math import hypot, isfinite
from statistics import median

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..formulas import (
    _affine_point,
    _affine_size,
    _greedy_unique_match,
    _linear_ratio_size_error,
    _log_ratio_size_error,
)
from .template_runtime import (
    BoxOrientation,
    LayoutSlot,
    LayoutTemplate,
    get_layout_template,
)


@dataclass(frozen=True)
class TemplateObservation:
    """Rectangulo usado para matching, independiente del ROI.

    ``u/v/width/height`` son coordenadas crudas de imagen. Los campos
    canonicos opcionales se conservan solo para materializar la escena y la
    geometria del contador despues de decidir A/B.
    """

    u: float
    v: float
    width: float
    height: float
    canonical_u: float | None = None
    canonical_v: float | None = None
    canonical_width: float | None = None
    canonical_height: float | None = None

    @property
    def scene_rect(self) -> tuple[float, float, float, float]:
        return (
            self.u if self.canonical_u is None else self.canonical_u,
            self.v if self.canonical_v is None else self.canonical_v,
            self.width if self.canonical_width is None else self.canonical_width,
            self.height if self.canonical_height is None else self.canonical_height,
        )


@dataclass(frozen=True)
class TemplateAssignment:
    observation_index: int
    level: int
    slot: LayoutSlot
    error: float


@dataclass(frozen=True)
class TemplateFit:
    assignments: tuple[TemplateAssignment, ...]
    mean_error: float
    phase: int = 0  # patron usado en N0: 0=A, 1=B
    # Slots completos reconstruidos exclusivamente por fragmentos. El valor
    # por defecto mantiene compatibles todos los constructores existentes.
    inferred_slots: tuple[tuple[int, LayoutSlot], ...] = ()
    # sx, sy, tx, ty: registra el template normalizado en coordenadas de imagen.
    registration: tuple[float, float, float, float] | None = None


def transform_layout_template(
    template: LayoutTemplate,
    registration: tuple[float, float, float, float] | None,
) -> LayoutTemplate:
    """Superpone un template normalizado sobre la nube de cajas observada.

    La transformacion conserva IDs y topologia; solo adapta la geometria a
    coordenadas de imagen. El ROI no participa en este calculo.
    """
    if registration is None:
        return template
    sx, sy, tx, ty = registration

    def transform_slot(slot: LayoutSlot) -> LayoutSlot:
        u, v = _affine_point(slot.u, slot.v, sx, sy, tx, ty)
        width, height = _affine_size(slot.width, slot.height, sx, sy)
        return LayoutSlot(
            cell=slot.cell, u=u, v=v, width=width, height=height,
            orientation=slot.orientation,
        )

    return LayoutTemplate(
        box_class=template.box_class,
        pattern=template.pattern,
        slots=tuple(transform_slot(slot) for slot in template.slots),
    )


def _orientation(width: float, height: float) -> BoxOrientation:
    return BoxOrientation.HORIZONTAL if width >= height else BoxOrientation.VERTICAL


def _estimate_registration(
    templates: list[tuple[int, LayoutTemplate]],
    observations: tuple[TemplateObservation, ...],
    max_center_distance: float,
) -> tuple[float, float, float, float] | None:
    """Estima escala XY + traslacion sin usar los bordes del ROI.

    El tamano mediano de las cajas fija la escala. Luego un voto entre pares
    observacion/slot encuentra la traslacion que explica mas cajas unicas.
    Esto permite registrar desde una muestra parcial sin exigir que la carga
    toque los cuatro extremos de la paleta.
    """
    slots = [slot for _level, template in templates for slot in template.slots]
    if not observations or not slots:
        return None

    x_ratios: list[float] = []
    y_ratios: list[float] = []
    for orientation in BoxOrientation:
        observed = [item for item in observations if _orientation(item.width, item.height) is orientation]
        expected = [slot for slot in slots if slot.orientation is orientation]
        if not observed or not expected:
            continue
        expected_width = median(slot.width for slot in expected)
        expected_height = median(slot.height for slot in expected)
        x_ratios.extend(item.width / expected_width for item in observed)
        y_ratios.extend(item.height / expected_height for item in observed)
    if not x_ratios or not y_ratios:
        return None
    sx, sy = median(x_ratios), median(y_ratios)
    # En coordenadas crudas la escala depende de la resolucion y normalmente
    # es mucho mayor que 1. Solo se exige que sea valida y positiva; limitarla
    # al rango normalizado 0.25..4 descartaba toda captura real en pixeles.
    if not (isfinite(sx) and isfinite(sy) and sx > 0.0 and sy > 0.0):
        return None

    candidates: list[tuple[float, float]] = []
    for item in observations:
        orientation = _orientation(item.width, item.height)
        for slot in slots:
            if slot.orientation is orientation:
                candidates.append((item.u - sx * slot.u, item.v - sy * slot.v))

    registered_center_distance = max_center_distance * max(sx, sy)

    def score(offset: tuple[float, float]) -> tuple[int, float]:
        tx, ty = offset
        edges: list[tuple[float, int, int]] = []
        for obs_index, item in enumerate(observations):
            orientation = _orientation(item.width, item.height)
            for slot_index, slot in enumerate(slots):
                if slot.orientation is not orientation:
                    continue
                distance = hypot(item.u - (sx * slot.u + tx), item.v - (sy * slot.v + ty))
                if distance <= registered_center_distance:
                    edges.append((distance, obs_index, slot_index))
        return _greedy_unique_match(edges)

    if not candidates:
        return None
    scored = [(score(item), item) for item in set(candidates)]
    (_best_count, _best_error), (tx, ty) = max(
        scored,
        key=lambda item: (item[0][0], -item[0][1]),
    )
    matched, _error = score((tx, ty))
    if matched < min(len(observations), 3):
        return None
    return sx, sy, tx, ty


def match_layout_slot(
    template: LayoutTemplate,
    center: tuple[float, float],
    footprint: tuple[float, float],
    occupied: set[int],
    max_center_distance: float,
) -> LayoutSlot | None:
    """Asigna una observacion al hueco libre compatible mas cercano."""
    u, v = center
    width, height = footprint
    observed_orientation = (
        BoxOrientation.HORIZONTAL if width >= height else BoxOrientation.VERTICAL
    )
    candidates: list[tuple[float, LayoutSlot]] = []
    for slot in template.slots:
        if slot.orientation is not observed_orientation:
            continue
        center_error = hypot(u - slot.u, v - slot.v)
        if center_error > max_center_distance:
            continue
        size_error = _linear_ratio_size_error(width, height, slot.width, slot.height)
        candidates.append((center_error + 0.05 * size_error, slot))
    if not candidates:
        return None
    matched = min(candidates, key=lambda item: (item[0], item[1].cell))[1]
    # Identificar primero el hueco evita desplazar una redeteccion ocupada a
    # una posicion vecina libre dentro de una tolerancia amplia.
    return None if matched.cell in occupied else matched


def _observation_slot_error(
    observation: TemplateObservation,
    slot: LayoutSlot,
    max_center_distance: float,
    min_side_ratio: float,
    *,
    size_correction: tuple[float, float] = (1.0, 1.0),
) -> float | None:
    orientation = (
        BoxOrientation.HORIZONTAL
        if observation.width >= observation.height
        else BoxOrientation.VERTICAL
    )
    if orientation is not slot.orientation:
        return None
    center_error = hypot(observation.u - slot.u, observation.v - slot.v)
    if center_error > max_center_distance:
        return None
    width_ratio = observation.width / max(slot.width * size_correction[0], 1e-9)
    height_ratio = observation.height / max(slot.height * size_correction[1], 1e-9)
    max_side_ratio = 1.0 / max(min_side_ratio, 1e-9)
    if not (
        min_side_ratio <= width_ratio <= max_side_ratio
        and min_side_ratio <= height_ratio <= max_side_ratio
    ):
        return None
    size_error = _log_ratio_size_error(width_ratio, height_ratio)
    return center_error + 0.05 * size_error


def fit_layout_hypothesis(
    box_class: str,
    base_level: int,
    observations: tuple[TemplateObservation, ...],
    include_upper: bool,
    max_center_distance: float,
    min_side_ratio: float,
    phase: int = 0,
    max_assignment_states: int | None = None,
) -> TemplateFit | None:
    """Ajuste global uno-a-uno contra una o dos plantillas alternantes.

    La asignacion se resuelve como matching bipartito de costo minimo. No se
    enumeran permutaciones: una escena dudosa termina en tiempo polinomial y
    devuelve ``None`` si no existe matching completo.
    """
    lower = get_layout_template(box_class, base_level, phase)
    if lower is None:
        return None
    templates = [(base_level, lower)]
    if include_upper:
        upper = get_layout_template(box_class, base_level + 1, phase)
        if upper is None:
            return None
        templates.append((base_level + 1, upper))

    slot_refs = [
        (level, slot)
        for level, template in templates
        for slot in template.slots
    ]
    if len(observations) > len(slot_refs):
        return None

    registration = _estimate_registration(
        templates, observations, max_center_distance,
    )
    registered_center_distance = max_center_distance
    if registration is not None:
        templates = [
            (level, transform_layout_template(template, registration))
            for level, template in templates
        ]
        registered_center_distance *= max(registration[0], registration[1])

    # Una unica escala XY puede registrar los centros, pero no siempre los
    # footprints bajo perspectiva. Si horizontales y verticales exigen escalas
    # incompatibles entre si, se corrige el tamano por orientacion; el gate de
    # lados permanece activo para que un fragmento no pase como caja completa.
    size_corrections = {orientation: (1.0, 1.0) for orientation in BoxOrientation}
    orientation_scales: dict[BoxOrientation, tuple[float, float]] = {}
    for orientation in BoxOrientation:
        observed = [item for item in observations if _orientation(item.width, item.height) is orientation]
        expected = [
            slot
            for _level, template in templates
            for slot in template.slots
            if slot.orientation is orientation
        ]
        if observed and expected:
            orientation_scales[orientation] = (
                median(item.width for item in observed) / median(slot.width for slot in expected),
                median(item.height for item in observed) / median(slot.height for slot in expected),
            )
    # Las bolsas son deformables y su bbox cambia mucho segun orientacion y
    # profundidad. Los rollos rigidos conservan el gate global original, que
    # ademas evita confundir dos fragmentos con evidencia de otra capa.
    if box_class.startswith("bag_") and len(orientation_scales) > 1:
        max_side_ratio = 1.0 / max(min_side_ratio, 1e-9)
        perspective_detected = any(
            max(values) / max(min(values), 1e-9) > max_side_ratio
            for values in zip(*orientation_scales.values())
        )
        if perspective_detected:
            size_corrections.update(orientation_scales)

    candidates_by_observation: list[list[tuple[float, int, LayoutSlot]]] = []
    for observation in observations:
        candidates: list[tuple[float, int, LayoutSlot]] = []
        for level, template in templates:
            for slot in template.slots:
                error = _observation_slot_error(
                    observation, slot, registered_center_distance, min_side_ratio,
                    size_correction=size_corrections[_orientation(
                        observation.width, observation.height,
                    )],
                )
                if error is not None:
                    candidates.append((error, level, slot))
        if not candidates:
            return None
        candidates_by_observation.append(candidates)

    slot_index = {
        (level, slot.cell): index
        for index, (level, slot) in enumerate(slot_refs)
    }
    impossible = 1e12
    costs = np.full((len(observations), len(slot_refs)), impossible, dtype=float)
    for observation_index, candidates in enumerate(candidates_by_observation):
        for error, level, slot in candidates:
            costs[observation_index, slot_index[(level, slot.cell)]] = error

    def solve_cost_matrix(
        forced: tuple[int, int] | None = None,
    ) -> tuple[float, tuple[TemplateAssignment, ...]] | None:
        forced_rows = set() if forced is None else {forced[0]}
        forced_cols = set() if forced is None else {forced[1]}
        rows = [index for index in range(len(observations)) if index not in forced_rows]
        cols = [index for index in range(len(slot_refs)) if index not in forced_cols]
        if len(rows) > len(cols):
            return None
        selected: list[tuple[int, int]] = [] if forced is None else [forced]
        if rows:
            row_indices, col_indices = linear_sum_assignment(
                costs[np.ix_(rows, cols)],
            )
            selected.extend(
                (rows[int(row_index)], cols[int(col_index)])
                for row_index, col_index in zip(row_indices, col_indices)
            )
        if any(costs[row, col] >= impossible for row, col in selected):
            return None
        assignments = tuple(
            TemplateAssignment(
                observation_index=row,
                level=slot_refs[col][0],
                slot=slot_refs[col][1],
                error=float(costs[row, col]),
            )
            for row, col in sorted(selected)
        )
        return sum(item.error for item in assignments), assignments

    solved = solve_cost_matrix()
    if solved is None:
        return None
    if include_upper and not any(item.level != base_level for item in solved[1]):
        # La hipotesis de dos niveles exige al menos una caja arriba. Si el
        # optimo libre usa solo N0, se fuerza cada arista N1 posible y se toma
        # el mejor matching restante. Sigue siendo polinomial y acotado.
        forced_solutions = []
        for observation_index, candidates in enumerate(candidates_by_observation):
            for _error, level, slot in candidates:
                if level == base_level:
                    continue
                candidate = solve_cost_matrix((
                    observation_index,
                    slot_index[(level, slot.cell)],
                ))
                if candidate is not None:
                    forced_solutions.append(candidate)
        if not forced_solutions:
            return None
        solved = min(forced_solutions, key=lambda item: item[0])
    assignments = solved[1]
    error_scale = (
        max(registration[0], registration[1])
        if registration is not None
        else 1.0
    )
    return TemplateFit(
        assignments=assignments,
        mean_error=solved[0] / max(len(assignments), 1) / error_scale,
        phase=phase,
        registration=registration,
    )


__all__ = [
    "TemplateAssignment",
    "TemplateFit",
    "TemplateObservation",
    "fit_layout_hypothesis",
    "match_layout_slot",
    "transform_layout_template",
]
