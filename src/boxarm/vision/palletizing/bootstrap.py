from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm -- ver header completo en counter.py (paquete
# vision/palletizing/, historia acumulativa unica del modulo original).
# -----------------------------------------------------------------------
"""Mixin _BootstrapMixin: reconciliacion combinatoria del inventario
inicial por fronteras consecutivas Ni/N(i+1) antes del primer ciclo de
brazo, y los pares de oclusion 2D que la alimentan."""

import itertools
import logging

import numpy as np

from .formulas import (
    _bbox_center_and_size,
    _bbox_max_side,
    _bootstrap_bbox_text,
    _center_distance_over_min_side,
    _integral_rect_sum,
    _intersection_over_min_area,
    _interval_samples,
    _is_duplicate_observation,
    _observed_median,
    _partial_fit_slack,
    _project,
    _quantize_position,
    _rasterize_rect,
    _rect_overlap_over_min,
    _rect_overflows_unit_square,
    _rect_union_coverage,
    _support_polygon_assessment,
    _template_min_evidence,
)
from .templates.template_matcher import (
    TemplateFit,
    TemplateObservation,
    fit_layout_hypothesis,
    transform_layout_template,
)
from .templates.template_runtime import get_layout_template, get_template_capacity
from .types import ParsedDetection

logger = logging.getLogger(__name__)


class _RectUnionCoverageIndex:
    """Indice integral de la union rasterizada de rectangulos canonicos.

    ``_rect_union_coverage`` reconstruye una mascara ``n x n`` para cada
    consulta. Durante el bootstrap, sin embargo, cientos de candidatos se
    contrastan contra el MISMO conjunto de cajas superiores. Este indice
    rasteriza esa union una vez y conserva exactamente los mismos limites de
    pixel de la formula F7; cada cobertura posterior cuesta cuatro lecturas.
    """

    def __init__(
        self,
        rects: list[tuple[float, float, float, float]],
        occupancy_grid: int,
    ) -> None:
        self._size = occupancy_grid
        mask = np.zeros((occupancy_grid, occupancy_grid), dtype=bool)
        for u, v, du, dv in rects:
            u0, u1, v0, v1 = _rasterize_rect(u, v, du, dv, occupancy_grid)
            if u1 > u0 and v1 > v0:
                mask[v0:v1, u0:u1] = True

        # El borde cero permite obtener la suma de cualquier ventana con la
        # identidad de imagen integral sin condiciones especiales.
        self._integral = np.zeros(
            (occupancy_grid + 1, occupancy_grid + 1), dtype=np.int32,
        )
        self._integral[1:, 1:] = mask.cumsum(
            axis=0, dtype=np.int32,
        ).cumsum(axis=1, dtype=np.int32)

    def coverage(self, target: tuple[float, float, float, float]) -> float:
        u, v, du, dv = target
        u0, u1, v0, v1 = _rasterize_rect(u, v, du, dv, self._size)
        if u1 <= u0 or v1 <= v0:
            return 0.0
        covered = _integral_rect_sum(self._integral, u0, u1, v0, v1)
        return float(covered) / float((u1 - u0) * (v1 - v0))


class _BootstrapMixin:
    @staticmethod
    def _deduplicate_template_observations(
        observations: list[TemplateObservation],
    ) -> tuple[TemplateObservation, ...]:
        """Elimina bboxes duplicados antes del ajuste global del template.

        Dos niveles reales pueden compartir centro, pero cambian de escala por
        perspectiva. Solo se fusionan observaciones casi coincidentes Y de
        tamaño equivalente; se conserva la de mayor area.
        """
        unique: list[TemplateObservation] = []
        for item in sorted(observations, key=lambda value: value.width * value.height, reverse=True):
            is_duplicate = any(
                _is_duplicate_observation(
                    (item.u, item.v), (item.width, item.height),
                    (kept.u, kept.v), (kept.width, kept.height),
                )
                for kept in unique
            )
            if not is_duplicate:
                unique.append(item)
        return tuple(unique)

    """Reconciliacion del inventario inicial y pares de oclusion 2D."""

    @staticmethod
    def _template_bootstrap_signature(
        fit: TemplateFit,
        observations: tuple[TemplateObservation, ...],
        base_level: int,
    ) -> tuple:
        """Identidad temporal completa de una hipotesis de plantilla.

        Los slots reconstruidos con fragmentos son parte de la hipotesis. Si
        cambian entre frames, la racha debe reiniciarse aunque las cajas
        completas sigan asignadas a los mismos slots.
        """
        assignments = tuple(sorted(
            (item.level, item.slot.cell) for item in fit.assignments
        ))
        inferred = tuple(sorted(
            (level, slot.cell) for level, slot in fit.inferred_slots
        ))
        return (
            "template", base_level, fit.phase, assignments, inferred,
        )

    def _template_partials_are_explained(
        self,
        fit: TemplateFit,
        observations: tuple[TemplateObservation, ...],
        partials: tuple[TemplateObservation, ...],
        base_level: int,
    ) -> bool:
        return self._explain_two_level_partials(
            fit, observations, partials, base_level,
        ) is not None

    def _explain_two_level_partials(
        self,
        fit: TemplateFit,
        observations: tuple[TemplateObservation, ...],
        partials: tuple[TemplateObservation, ...],
        base_level: int,
    ) -> TemplateFit | None:
        """Comprueba que cada fragmento cabe abajo de una caja superior.

        Un bbox pequeno por si solo no prueba otro piso. Debe estar contenido
        en algun rectangulo del patron inferior y ese rectangulo completo
        debe quedar ocluido por una observacion asignada al patron superior.
        """
        if not partials:
            return fit
        lower = get_layout_template(
            self._box_class or "", base_level, fit.phase,
        )
        if lower is None:
            return None
        lower = transform_layout_template(lower, fit.registration)
        upper_rects = [
            (
                observations[item.observation_index].u,
                observations[item.observation_index].v,
                observations[item.observation_index].width,
                observations[item.observation_index].height,
            )
            for item in fit.assignments
            if item.level == base_level + 1
        ]
        if not upper_rects:
            return None
        assigned_lower = {
            item.slot.cell
            for item in fit.assignments
            if item.level == base_level
        }
        inferred = {
            (level, slot.cell): (level, slot)
            for level, slot in fit.inferred_slots
        }
        for partial in partials:
            partial_rect = (partial.u, partial.v, partial.width, partial.height)
            compatible: list[tuple[float, object]] = []
            for slot in lower.slots:
                lower_rect = (slot.u, slot.v, slot.width, slot.height)
                overlap = _rect_overlap_over_min(partial_rect, lower_rect)
                if (
                    partial.width > slot.width * 1.25
                    or partial.height > slot.height * 1.25
                    or overlap < self._cfg.tau_overlap
                ):
                    continue
                if any(
                    _rect_overlap_over_min(lower_rect, upper_rect)
                    > self._cfg.max_same_level_overlap
                    for upper_rect in upper_rects
                ):
                    compatible.append((overlap, slot))
            if not compatible:
                return None
            _overlap, matched = max(
                compatible,
                key=lambda item: (item[0], -item[1].cell),
            )
            if matched.cell not in assigned_lower:
                inferred[(base_level, matched.cell)] = (base_level, matched)
        return TemplateFit(
            assignments=fit.assignments,
            mean_error=fit.mean_error,
            phase=fit.phase,
            inferred_slots=tuple(inferred.values()),
            registration=fit.registration,
        )

    def _explain_one_level_partials(
        self,
        fit: TemplateFit,
        partials: tuple[TemplateObservation, ...],
        base_level: int,
    ) -> TemplateFit | None:
        """Explica fragmentos como duplicados o como un slot libre reconstruido.

        La asignacion de completas ya es inyectiva. Los fragmentos pueden
        compartir slot porque dos recortes complementarios representan una
        sola caja; se valida la UNION geometrica, nunca la suma de sus lados.
        """
        if not partials:
            return fit
        template = get_layout_template(
            self._box_class or "", base_level, fit.phase,
        )
        if template is None:
            return None
        template = transform_layout_template(template, fit.registration)
        occupied = {
            assignment.slot.cell
            for assignment in fit.assignments
            if assignment.level == base_level
        }

        def rect(item: TemplateObservation | object):
            return (item.u, item.v, item.width, item.height)

        groups: dict[int, list[tuple[float, float, float, float]]] = {}
        for partial in partials:
            partial_rect = rect(partial)
            compatible = []
            for slot in template.slots:
                slot_rect = rect(slot)
                if (
                    partial.width > slot.width * 1.25
                    or partial.height > slot.height * 1.25
                    or _rect_overlap_over_min(partial_rect, slot_rect)
                    < self._cfg.tau_overlap
                ):
                    continue
                compatible.append((
                    _rect_overlap_over_min(partial_rect, slot_rect), slot,
                ))
            if not compatible:
                return None
            compatible.sort(key=lambda item: (-item[0], item[1].cell))
            best_overlap, best_slot = compatible[0]
            # Un recorte dentro de una completa ya asignada es una deteccion
            # duplicada, no evidencia de otro nivel ni de otra caja.
            if best_slot.cell in occupied:
                continue
            # Si dos slots libres explican igual el recorte, no se inventa la
            # identidad: otro frame debera desambiguarlo.
            if (
                len(compatible) > 1
                and compatible[1][0] >= best_overlap - self._cfg.partial_fit_tolerance
            ):
                return None
            groups.setdefault(best_slot.cell, []).append(partial_rect)

        inferred = []
        min_coverage = self._cfg.min_complete_side_ratio
        by_cell = {slot.cell: slot for slot in template.slots}
        for cell, fragments in groups.items():
            slot = by_cell[cell]
            coverage = _rect_union_coverage(
                rect(slot), fragments, self._cfg.occupancy_grid,
            )
            if coverage < min_coverage:
                return None
            inferred.append((base_level, slot))
        return TemplateFit(
            assignments=fit.assignments,
            mean_error=fit.mean_error,
            phase=fit.phase,
            inferred_slots=tuple(sorted(inferred, key=lambda item: item[1].cell)),
            registration=fit.registration,
        )

    def _select_template_bootstrap_fit(
        self,
        observations: tuple[TemplateObservation, ...],
        partials: tuple[TemplateObservation, ...],
        base_level: int,
    ) -> TemplateFit | None:
        """Elige una o dos capas por validez global y error geometrico."""
        min_side_ratio = self._cfg.min_complete_side_ratio
        margin = max(
            self._cfg.partial_fit_tolerance,
            1.0 / max(self._cfg.occupancy_grid, 1),
        )

        def unique_best(candidates: list[TemplateFit]) -> TemplateFit | None:
            ranked = sorted(candidates, key=lambda item: item.mean_error)
            if not ranked:
                return None
            if (
                len(ranked) > 1
                and ranked[1].mean_error - ranked[0].mean_error <= margin
            ):
                return None
            return ranked[0]

        template = get_layout_template(self._box_class or "", base_level)
        capacity = len(template.slots) if template is not None else 0
        force_two_levels = len(observations) + len(partials) >= capacity + 2
        one_candidates = [] if force_two_levels else [
            fit
            for phase in (0, 1)
            if (fit := fit_layout_hypothesis(
                self._box_class or "", base_level, observations, False,
                self._cfg.tau_cell, min_side_ratio, phase=phase,
            )) is not None
        ]
        # Una escena con 28 posiciones visibles y capacidad 25 no contiene
        # "28 cajas arriba": mezcla N0 visible/parcial con N1. La hipotesis
        # de dos niveles es obligatoria para explicar esa oclusion.
        two_candidates = [
            fit
            for phase in (0, 1)
            if (fit := fit_layout_hypothesis(
                self._box_class or "", base_level, observations, True,
                self._cfg.tau_cell, min_side_ratio, phase=phase,
            )) is not None
        ]
        explained_one = [
            explained
            for fit in one_candidates
            if (explained := self._explain_one_level_partials(
                fit, partials, base_level,
            )) is not None
        ]
        explained_two = [
            explained
            for fit in two_candidates
            if (explained := self._explain_two_level_partials(
                fit, observations, partials, base_level,
            )) is not None
        ]
        # En dos capas, el piso inferior precede fisicamente al superior. Si
        # A/B producen el mismo error al intercambiarse, conserva la hipotesis
        # con mayor evidencia completa abajo; evita un empate artificial de
        # fase para, por ejemplo, 15 cajas inferiores + 1 superior.
        if explained_two:
            lower_counts = [
                sum(item.level == base_level for item in fit.assignments)
                for fit in explained_two
            ]
            most_supported_lower = max(lower_counts)
            explained_two = [
                fit for fit, count in zip(explained_two, lower_counts)
                if count == most_supported_lower
            ]
        self._last_template_fit_diagnostics = (
            len(one_candidates), len(two_candidates),
            len(explained_one), len(explained_two),
        )
        one_level = unique_best(explained_one)
        two_levels = unique_best(explained_two)
        if one_level is None:
            return two_levels
        if two_levels is None:
            return one_level
        # Principio de parsimonia: no se inventa otro piso por una mejora
        # numerica diminuta. Dos niveles ganan solo si reducen el error por
        # mas que la incertidumbre geometrica de reconstruccion.
        if two_levels.mean_error + margin < one_level.mean_error:
            return two_levels
        return one_level

    def _apply_template_bootstrap_fit(
        self,
        fit: TemplateFit,
        observations: tuple[TemplateObservation, ...],
        base_level: int,
    ) -> None:
        """Materializa un ajuste global validado contra los dos patrones.

        Si aparece al menos una caja del nivel superior, la regla fisica de
        paletizado prueba que el inferior ya estaba completo. Sus huecos no
        visibles se reconstruyen desde la plantilla, no mediante packing
        libre. Las observadas conservan su geometria medida.
        """
        upper_level = base_level + 1
        has_upper = any(item.level == upper_level for item in fit.assignments)
        lower_template = get_layout_template(
            self._box_class or "", base_level, fit.phase,
        )
        if lower_template is None:
            raise ValueError("no existe plantilla inferior para aplicar el bootstrap")
        lower_template = transform_layout_template(
            lower_template, fit.registration,
        )

        def canonical_slot(slot):
            u, v = _project(self._homography, slot.u, slot.v)
            width, height = self._measure_footprint(
                slot.u, slot.v, slot.width, slot.height,
            )
            return u, v, width, height

        targets: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        # Los rollos se cargan por capas completas. Las bolsas pueden apoyar
        # una capa superior antes de completar todos los huecos inferiores;
        # en ese caso solo se materializa lo observado o reconstruido desde
        # fragmentos, nunca la capacidad entera por suposicion.
        if has_upper and not (self._box_class or "").startswith("bag_"):
            targets.update({
                (slot.cell, base_level): canonical_slot(slot)
                for slot in lower_template.slots
            })
        for assignment in fit.assignments:
            observed = observations[assignment.observation_index]
            targets[(assignment.slot.cell, assignment.level)] = observed.scene_rect
        for level, slot in fit.inferred_slots:
            targets[(slot.cell, level)] = canonical_slot(slot)

        affected_levels = {base_level, upper_level}
        previous_total = self.total
        for key in [key for key in self._occupied if key[1] in affected_levels]:
            self._occupied.remove(key)
            self._dynamic_positions.pop(key, None)
            self._footprint.pop(key, None)
            self._cell_frame.pop(key, None)

        for key, (u, v, width, height) in targets.items():
            self._occupied.add(key)
            self._dynamic_positions[key] = (u, v)
            self._footprint[key] = (width, height)
            self._cell_frame[key] = self._current_frame

        for level in affected_levels:
            cells = [cell for cell, z in self._occupied if z == level]
            self._next_cell_by_level[level] = max(cells, default=-1) + 1
            self._recompute_level_footprint(level)

        self.total = len(self._occupied)
        inferred = max(0, self.total - previous_total)
        self.initial += inferred
        if self.initial + self.placed != self.total:
            self.initial = max(0, self.total - self.placed)
        lower_is_full = (
            sum(1 for _cell, level in self._occupied if level == base_level)
            == len(lower_template.slots)
        )
        if has_upper and lower_is_full:
            self._proven_full.add(base_level)
        self._template_phase = fit.phase
        self._template_registration = fit.registration

        logger.debug(
            "bootstrap por plantillas: fase=N0=%s hipotesis=%s error_medio=%.4f "
            "N%d=%d N%d=%d ocultas_inferidas=%d",
            "A" if fit.phase == 0 else "B",
            "dos-niveles" if has_upper else "un-nivel", fit.mean_error,
            base_level, sum(1 for _g, z in self._occupied if z == base_level),
            upper_level, sum(1 for _g, z in self._occupied if z == upper_level),
            inferred,
        )
        self._finish_initial_bootstrap(
            upper_level if has_upper else base_level,
            "ajuste global estable contra plantillas alternantes A/B",
        )

    def _bootstrap_partial_hypotheses(
        self,
        u: float,
        v: float,
        measured: tuple[float, float],
        level: int,
        occupied_rects: dict[int, tuple[float, float, float, float]],
    ) -> list[tuple[tuple[float, float, float, float], frozenset[int], float]]:
        """Colocaciones de un fragmento y celdas que tendrian que subir.

        Una caja superior mal clasificada como N0 aparece como interpenetracion
        al completar el fragmento inferior. Durante el bootstrap esa colision
        no se descarta inmediatamente: se convierte en la hipotesis de que la
        celda conflictiva pertenece a N1.
        """
        consensus = self._level_footprint.get(level)
        if consensus is None:
            return []
        long_side, short_side = consensus
        du, dv = measured
        tolerance = self._cfg.partial_fit_tolerance
        limit = self._cfg.max_same_level_overlap
        grouped: dict[
            tuple[bool, tuple[int, ...], tuple[int, int, bool]],
            tuple[tuple[float, float, float, float], frozenset[int], float],
        ] = {}
        quantum = max(tolerance, 0.01)
        for width, height in ((long_side, short_side), (short_side, long_side)):
            if du > width * 1.25 or dv > height * 1.25:
                continue
            slack_u = _partial_fit_slack(width, du, tolerance)
            slack_v = _partial_fit_slack(height, dv, tolerance)
            for cu in _interval_samples(u, slack_u):
                for cv in _interval_samples(v, slack_v):
                    rect = (cu, cv, width, height)
                    if _rect_overflows_unit_square(cu, cv, width, height):
                        continue
                    overlaps = {
                        cell: _rect_overlap_over_min(rect, other)
                        for cell, other in occupied_rects.items()
                    }
                    conflicts = frozenset(cell for cell, ratio in overlaps.items() if ratio > limit)
                    if not conflicts:
                        continue
                    residual = max(
                        (ratio for cell, ratio in overlaps.items() if cell not in conflicts),
                        default=0.0,
                    )
                    if residual > limit:
                        continue
                    cover = _rect_union_coverage(
                        rect, [occupied_rects[cell] for cell in conflicts],
                        self._cfg.occupancy_grid,
                    )
                    # No colapsar todos los centros que chocan con las mismas
                    # celdas. El centro de mayor cobertura LOCAL puede volver
                    # incompatibles dos reconstrucciones entre si, aunque
                    # exista otra colocacion del mismo intervalo que resuelva
                    # el inventario GLOBAL. La equivalencia final ya agrupa
                    # despues las variantes numericas de una misma solucion.
                    key = (
                        width >= height,
                        tuple(sorted(conflicts)),
                        _quantize_position(cu, cv, quantum, width >= height),
                    )
                    candidate = (rect, conflicts, cover)
                    if key not in grouped or cover > grouped[key][2]:
                        grouped[key] = candidate
        return sorted(grouped.values(), key=lambda item: (-item[2], len(item[1])))

    def _bootstrap_hidden_candidates(
        self,
        lower: list[tuple[float, float, float, float]],
        upper: list[tuple[float, float, float, float]],
        footprint: tuple[float, float],
        level: int,
    ) -> list[tuple[float, float, float, float]]:
        """Posiciones canonicas que pueden estar ocultas bajo `upper`."""
        long_side, short_side = footprint
        limit = self._cfg.max_same_level_overlap
        candidates: dict[tuple[int, int, bool], tuple[float, tuple[float, float, float, float]]] = {}
        seeds = lower + upper
        quantum = max(self._cfg.partial_fit_tolerance, 0.01)
        upper_coverage = _RectUnionCoverageIndex(
            upper, self._cfg.occupancy_grid,
        )
        for width, height in ((long_side, short_side), (short_side, long_side)):
            us = {width / 2.0, 1.0 - width / 2.0}
            vs = {height / 2.0, 1.0 - height / 2.0}
            for su, sv, sdu, sdv in seeds:
                us.update((su, su - (sdu + width) / 2.0, su + (sdu + width) / 2.0,
                           su - width / 2.0, su + width / 2.0))
                vs.update((sv, sv - (sdv + height) / 2.0, sv + (sdv + height) / 2.0,
                           sv - height / 2.0, sv + height / 2.0))

            # Una caja inferior completamente oculta por dos cajas superiores
            # puede quedar centrada ENTRE ambas. Ese centro no coincide
            # necesariamente con ningun centro ni borde individual (caso
            # vertical oculto bajo dos cajas horizontales consecutivas), por
            # lo que el muestreo anterior nunca podia proponerlo. Los puntos
            # medios se derivan solo de pares superiores: son precisamente la
            # evidencia que debe cubrir al candidato y evitan ampliar el
            # dominio con combinaciones arbitrarias de cajas del mismo nivel.
            for first, second in itertools.combinations(upper, 2):
                us.add((first[0] + second[0]) / 2.0)
                vs.add((first[1] + second[1]) / 2.0)
            for cu in us:
                for cv in vs:
                    rect = (cu, cv, width, height)
                    if _rect_overflows_unit_square(cu, cv, width, height):
                        continue
                    if max((_rect_overlap_over_min(rect, other) for other in lower), default=0.0) > limit:
                        continue
                    hidden_share = upper_coverage.coverage(rect)
                    min_coverage, _max_ratio = self._support_threshold_values(
                        (width, height), level,
                    )
                    if hidden_share < min_coverage:
                        continue
                    key = _quantize_position(cu, cv, quantum, width >= height)
                    previous = candidates.get(key)
                    if previous is None or hidden_share > previous[0]:
                        candidates[key] = (hidden_share, rect)
        return [item[1] for item in sorted(candidates.values(), key=lambda item: -item[0])]

    def _bootstrap_top_is_supported(
        self,
        upper: tuple[float, float, float, float],
        lower: list[tuple[float, float, float, float]],
        level: int,
    ) -> bool:
        """Aplica en bootstrap el MISMO hull/fallback que durante operación.

        Tener dos criterios distintos permitía aceptar una configuración al
        reconstruirla y rechazarla en el frame siguiente. Aquí no hay estado
        temporal: todos los rectángulos de ``lower`` son soportes candidatos.
        """
        footprint = (upper[2], upper[3])
        relative_error = self._geometric_relative_error(footprint, level)
        assessment = _support_polygon_assessment(
            upper, lower,
            min_contact_ratio=0.5 * relative_error,
            min_hull_area_ratio=0.5 * relative_error,
            center_margin_ratio=0.0,
        )
        if assessment.interlocked:
            return True
        if assessment.contact_count < 2 or not assessment.degenerate:
            return False

        min_coverage, max_ratio = self._support_threshold_values(footprint, level)
        coverage = _rect_union_coverage(upper, lower, self._cfg.occupancy_grid)
        return (
            coverage >= min_coverage
            and self._dynamic_support_is_balanced(
                list(assessment.shares), min_coverage, max_ratio,
            )
        )

    def _bootstrap_rects_aligned(
        self,
        reference: list[tuple[float, float, float, float]],
        candidate: list[tuple[float, float, float, float]],
        *,
        ignore_center: bool = False,
        reference_tags: list[tuple[int, ...]] | None = None,
        candidate_tags: list[tuple[int, ...]] | None = None,
    ) -> list[tuple[float, float, float, float]] | None:
        """Alinea dos colecciones que describen las mismas cajas fisicas.

        El solver genera varios centros dentro del mismo intervalo factible.
        No son soluciones fisicas distintas si conservan orientacion y cada
        centro difiere como maximo dos holguras de reconstruccion.
        """
        if len(reference) != len(candidate):
            return None
        if not reference:
            return []
        tolerance = max(2.0 * self._cfg.partial_fit_tolerance, 1e-6)
        adjacency = [
            [
                index
                for index, b in enumerate(candidate)
                if (
                (a[2] >= a[3]) == (b[2] >= b[3])
                and (ignore_center or abs(a[0] - b[0]) <= tolerance)
                and (ignore_center or abs(a[1] - b[1]) <= tolerance)
                and abs(a[2] - b[2]) <= tolerance
                and abs(a[3] - b[3]) <= tolerance
                and (
                    reference_tags is None
                    or candidate_tags is None
                    or reference_tags[reference_index] == candidate_tags[index]
                )
                )
            ]
            for reference_index, a in enumerate(reference)
        ]
        matched_reference_by_candidate: dict[int, int] = {}

        def assign(reference_index: int, seen: set[int]) -> bool:
            for candidate_index in adjacency[reference_index]:
                if candidate_index in seen:
                    continue
                seen.add(candidate_index)
                previous = matched_reference_by_candidate.get(candidate_index)
                if previous is None or assign(previous, seen):
                    matched_reference_by_candidate[candidate_index] = reference_index
                    return True
            return False

        if not all(assign(index, set()) for index in range(len(reference))):
            return None
        aligned: list[tuple[float, float, float, float] | None] = [None] * len(reference)
        for candidate_index, reference_index in matched_reference_by_candidate.items():
            aligned[reference_index] = candidate[candidate_index]
        if any(rect is None for rect in aligned):
            return None
        return [rect for rect in aligned if rect is not None]

    def _bootstrap_solution_groups(self, solutions: list[dict]) -> list[dict]:
        """Cocienta soluciones numericas por equivalencia geometrica.

        Devuelve una mediana observada por clase, no un promedio. Por ello las
        diez variantes del mismo hueco residual cuentan como UNA solucion.
        """
        groups: list[dict] = []
        for solution in solutions:
            for group in groups:
                reference = group["members"][0]
                if (
                    solution.get("base_level", 0) != reference.get("base_level", 0)
                    or solution["promoted"] != reference["promoted"]
                ):
                    continue
                aligned_partials = self._bootstrap_rects_aligned(
                    reference["partials"], solution["partials"],
                    ignore_center=True,
                    reference_tags=[tuple(c) for c in reference.get("partial_conflicts", [])],
                    candidate_tags=[tuple(c) for c in solution.get("partial_conflicts", [])],
                )
                aligned_hidden = self._bootstrap_rects_aligned(
                    reference["hidden"], solution["hidden"],
                )
                if aligned_partials is None or aligned_hidden is None:
                    continue
                group["members"].append({
                    "base_level": solution.get("base_level", 0),
                    "promoted": solution["promoted"],
                    "partials": aligned_partials,
                    "hidden": aligned_hidden,
                    "partial_conflicts": solution.get("partial_conflicts", []),
                })
                break
            else:
                groups.append({"members": [solution]})

        representatives: list[dict] = []
        for group in groups:
            members = group["members"]
            reference = members[0]

            def median_rects(name: str) -> list[tuple[float, float, float, float]]:
                return [
                    tuple(_observed_median(member[name][index][axis] for member in members)
                          for axis in range(4))
                    for index in range(len(reference[name]))
                ]

            representatives.append({
                "base_level": reference.get("base_level", 0),
                "promoted": reference["promoted"],
                "partials": median_rects("partials"),
                "hidden": median_rects("hidden"),
                "partial_conflicts": reference.get("partial_conflicts", []),
                "equivalent_variants": len(members),
            })
        return representatives

    def _bootstrap_solutions_equivalent(self, left: dict, right: dict) -> bool:
        """True si dos frames describen la misma composicion fisica inicial."""
        return (
            left.get("base_level", 0) == right.get("base_level", 0)
            and left["promoted"] == right["promoted"]
            and self._bootstrap_rects_aligned(left["partials"], right["partials"])
            is not None
            and self._bootstrap_rects_aligned(left["hidden"], right["hidden"])
            is not None
        )

    def _log_bootstrap_assignment(
        self,
        solution: dict,
        occupied_rects: dict[int, tuple[float, float, float, float]],
        phase: str,
    ) -> None:
        """Imprime cada bbox y el papel fisico validado por el bootstrap."""
        base_level = solution.get("base_level", 0)
        upper_level = base_level + 1
        promoted = solution["promoted"]
        lower_observed = [cell for cell in sorted(occupied_rects) if cell not in promoted]
        logger.debug(
            "bootstrap %s: ASIGNACION -- N%d=%d observadas + %d reconstruidas + "
            "%d oculta = %d; N%d=%d promovidas",
            phase, base_level, len(lower_observed), len(solution["partials"]), len(solution["hidden"]),
            len(lower_observed) + len(solution["partials"]) + len(solution["hidden"]),
            upper_level, len(promoted),
        )
        for cell in lower_observed:
            logger.debug(
                "bootstrap %s: box celda=%d rol=N%d_OBSERVADA %s",
                phase, cell, base_level, _bootstrap_bbox_text(occupied_rects[cell]),
            )
        for cell in sorted(promoted):
            logger.debug(
                "bootstrap %s: box celda=%d rol=N%d_PROMOVIDA %s",
                phase, cell, upper_level, _bootstrap_bbox_text(occupied_rects[cell]),
            )
        conflicts = solution.get("partial_conflicts", [])
        for index, rect in enumerate(solution["partials"]):
            conflict_text = sorted(conflicts[index]) if index < len(conflicts) else []
            logger.debug(
                "bootstrap %s: box parcial=%d rol=N%d_RECONSTRUIDA %s "
                "conflicta_si_N%d_con=%s",
                phase, index, base_level, _bootstrap_bbox_text(rect), base_level, conflict_text,
            )
        for index, rect in enumerate(solution["hidden"]):
            logger.debug(
                "bootstrap %s: box oculta=%d rol=N%d_OCULTA_RESIDUAL %s",
                phase, index, base_level, _bootstrap_bbox_text(rect),
            )
        logger.debug(
            "bootstrap %s: barrera fuerte -- mantener %s en N%d impide completar "
            "rectangulos canonicos disjuntos; esta hipotesis los promueve y "
            "completa N%d con capacidad exacta (solo se aplica si su clase "
            "fisica es unica y estable)",
            phase, sorted(promoted), base_level, base_level,
        )

    def _apply_bootstrap_solution(self, solution: dict) -> None:
        """Materializa una frontera unica y avanza el solver a la siguiente."""
        base_level = solution.get("base_level", self._bootstrap_level)
        upper_level = base_level + 1
        promoted: set[int] = solution["promoted"]
        partial_rects: list[tuple[float, float, float, float]] = solution["partials"]
        hidden_rects: list[tuple[float, float, float, float]] = solution["hidden"]

        for old_cell in sorted(promoted):
            old_key = (old_cell, base_level)
            pos = self._dynamic_positions.pop(old_key)
            fp = self._footprint.pop(old_key)
            frame = self._cell_frame.pop(old_key, self._current_frame)
            self._occupied.remove(old_key)
            new_cell = self._next_cell_by_level.get(upper_level, 0)
            self._next_cell_by_level[upper_level] = new_cell + 1
            new_key = (new_cell, upper_level)
            self._occupied.add(new_key)
            self._dynamic_positions[new_key] = pos
            self._footprint[new_key] = fp
            self._cell_frame[new_key] = frame

        added = 0
        for u, v, du, dv in partial_rects + hidden_rects:
            cell = self._next_cell_by_level.get(base_level, 0)
            self._next_cell_by_level[base_level] = cell + 1
            key = (cell, base_level)
            self._occupied.add(key)
            self._dynamic_positions[key] = (u, v)
            self._footprint[key] = (du, dv)
            self._cell_frame[key] = self._current_frame
            added += 1

        self.total += added
        self.initial += added
        self._proven_full.add(base_level)
        self._recompute_level_footprint(base_level)
        self._recompute_level_footprint(upper_level)
        self._bootstrap_consumed_partials.extend(self._last_bootstrap_partials)
        self._bootstrap_level = upper_level
        self._last_bootstrap_signature = None
        self._bootstrap_signature_hits = 0
        self._bootstrap_solution_history.clear()
        self._last_bootstrap_partials.clear()
        self._last_bootstrap_complete_count = 0
        logger.debug(
            "bootstrap: FRONTERA N%d->N%d aplicada -> N%d=%d "
            "(%d reconstruida(s), %d oculta(s)), N%d=%d; continua la frontera siguiente",
            base_level, upper_level, base_level,
            sum(1 for _g, z in self._occupied if z == base_level),
            len(partial_rects), len(hidden_rects), upper_level,
            sum(1 for _g, z in self._occupied if z == upper_level),
        )

    def _finish_initial_bootstrap(self, top_level: int, reason: str) -> None:
        """Publica el inventario solo cuando todas sus fronteras quedaron cerradas."""
        self._bootstrap_reconciled = True
        if self._template_baseline_pending:
            # Los ciclos ocurridos antes de poder leer una escena estable ya
            # estan contenidos en el snapshot aceptado. No son permisos para
            # agregar cajas futuras ni convierten la linea base en "placed".
            self._placement_credits = 0
            self._template_baseline_pending = False
        self._initial_scene_deferred = False
        self._last_bootstrap_signature = None
        self._bootstrap_signature_hits = 0
        self._bootstrap_solution_history.clear()
        self._last_bootstrap_partials.clear()
        self._last_bootstrap_complete_count = 0
        composition = {
            level: sum(1 for _cell, z in self._occupied if z == level)
            for level in range(top_level + 1)
        }
        logger.debug(
            "bootstrap: ESTADO INICIAL RECONCILIADO hasta N%d (%s) -> %s; "
            "el solver inicial queda desactivado y el ISO 3D ya puede publicarse",
            top_level, reason,
            ", ".join(f"N{level}={count}" for level, count in composition.items()),
        )
        self._advance_tracking_floor()
        self._log_overlapping_cells()
        self._log_cell_table()

    def _bootstrap_partial_was_consumed(
        self, rect: tuple[float, float, float, float],
    ) -> bool:
        """Evita reutilizar en Ni un recorte que ya reconstruyo N(i-1)."""
        return any(
            self._bootstrap_rects_aligned([previous], [rect]) is not None
            for previous in self._bootstrap_consumed_partials
        )

    def _reconcile_initial_layers(self, parsed: list[ParsedDetection]) -> None:
        """Resuelve sucesivamente cada frontera Ni -> N(i+1) del inventario.

        Ver el brazo durante el arranque NO termina esta fase. Puede entrar
        mientras la paleta inicial sigue ambigua; si se abandonara el solver
        en ese instante, ``_initial_scene_deferred`` quedaria activo para
        siempre y el ISO nunca publicaria los solidos. El unico cierre valido
        es una solucion geometrica estable o alcanzar el maximo de niveles.
        """
        if self._bootstrap_reconciled or self._cfg.layout_mode != "auto":
            return
        base_level = self._bootstrap_level
        if base_level >= self._cfg.levels - 1:
            self._finish_initial_bootstrap(base_level, "se alcanzo el maximo de niveles configurado")
            return
        # Para una clase calibrada, la plantilla es la fuente de verdad de la
        # capacidad.  Condicionar esta ruta a un numero duplicado en YAML
        # permitia caer silenciosamente al reconciliador geometrico generico
        # (y caro) cuando ambos valores faltaban o discrepaban.
        template = get_layout_template(self._box_class or "", base_level)
        capacity = (
            len(template.slots)
            if template is not None
            else get_template_capacity(self._box_class or "")
        )
        consensus = self._level_footprint.get(base_level)
        if capacity is None or consensus is None:
            return

        occupied_rects = {
            g: (*self._cell_position(g, base_level), *self._canonical_footprint((g, base_level)))
            for g, z in self._occupied
            if z == base_level and (g, z) in self._footprint
        }

        # NOTA: se penso en un atajo aqui comparando len(parsed) contra
        # `capacity` ANTES de calcular `partials` -- "si lo visible no
        # supera la capacidad, es solo N0". Se descarto: una caja realmente
        # apilada puede compartir area con una del piso (misma "capacidad"
        # visible en total) y ese atajo la fusionaba con la celda de abajo
        # por conteo bruto, sin mirar la evidencia geometrica real. Tampoco
        # basta la ausencia de parciales: el frame p4r1@02:20 contiene cajas
        # completas de dos niveles. Para clases calibradas, la autoridad es
        # el ajuste global de una/dos capas y ambas fases A/B.
        partials: list[tuple[float, float, tuple[float, float]]] = []
        template_partials: list[TemplateObservation] = []
        complete_observations: list[TemplateObservation] = []
        complete_count = 0
        calibrated_template = template
        for bbox, _conf, _cls_name in parsed:
            (cx, cy), (w_px, h_px) = _bbox_center_and_size(bbox)
            measured = self._measure_footprint(cx, cy, w_px, h_px)
            if calibrated_template is not None:
                # El consenso conserva los ejes canonicos (ancho, alto), no
                # garantiza orden largo/corto. Compararlo sin ordenar hacia
                # que una huella vertical clasificara casi todas las cajas
                # completas como fragmentos.
                expected_long, expected_short = max(consensus), min(consensus)
                observed_long, observed_short = max(measured), min(measured)
                side_ratio = self._cfg.min_complete_side_ratio
                is_partial = (
                    observed_long < expected_long * side_ratio
                    or observed_short < expected_short * side_ratio
                )
            else:
                is_partial = self._is_partial_footprint(measured, base_level)
            if is_partial:
                u, v = _project(self._homography, cx, cy)
                observed = (u, v, measured[0], measured[1])
                if not self._bootstrap_partial_was_consumed(observed):
                    partials.append((u, v, measured))
                    template_partials.append(TemplateObservation(
                        cx, cy, w_px, h_px,
                        canonical_u=u,
                        canonical_v=v,
                        canonical_width=measured[0],
                        canonical_height=measured[1],
                    ))
            else:
                complete_count += 1
                u, v = _project(self._homography, cx, cy)
                complete_observations.append(TemplateObservation(
                    cx, cy, w_px, h_px,
                    canonical_u=u,
                    canonical_v=v,
                    canonical_width=measured[0],
                    canonical_height=measured[1],
                ))

        if template is not None:
            partial_observations = tuple(template_partials)
            observations = self._deduplicate_template_observations(
                complete_observations,
            )
            evidence_count = len(observations) + len(partial_observations)
            # Con media plantilla o menos no hay evidencia suficiente para
            # fijar A/B. Esas cajas se publican normalmente, sin mostrar
            # "analizando": el template permanece abierto hasta observar
            # estrictamente mas de la mitad de la capacidad de ESTA clase.
            min_evidence = _template_min_evidence(capacity)
            if evidence_count < min_evidence:
                self._initial_scene_deferred = False
                self._bootstrap_signature_hits = 0
                self._last_bootstrap_signature = None
                logger.debug(
                    "bootstrap por plantillas: esperando evidencia inicial %d/%d "
                    "cajas o fragmentos antes de fijar la fase A/B",
                    evidence_count, min_evidence,
                )
                return
            self._initial_scene_deferred = True
            # Cache por entrada exacta: si la escena observada es identica a
            # la del frame anterior (video quieto, o el arm-gate esta en
            # pausa y no llegan detecciones nuevas), reusa el resultado en
            # vez de recorrer de nuevo el registro geometrico completo (ver
            # docstring de self._template_fit_cache_key en init_state.py).
            # Redondeo a 1e-4: mismo margen que ya usa el matching (ver
            # partial_fit_tolerance/tau_cell), dos detecciones del "mismo"
            # frame no coinciden bit a bit por ruido de punto flotante.
            def _round_obs(obs: tuple[TemplateObservation, ...]) -> tuple:
                return tuple(
                    (round(o.u, 4), round(o.v, 4), round(o.width, 4), round(o.height, 4))
                    for o in obs
                )
            fit_cache_key = (
                base_level, self._box_class, _round_obs(observations),
                _round_obs(partial_observations),
            )
            if fit_cache_key == self._template_fit_cache_key:
                fit = self._template_fit_cache_value
            else:
                fit = self._select_template_bootstrap_fit(
                    observations, partial_observations, base_level,
                )
                self._template_fit_cache_key = fit_cache_key
                self._template_fit_cache_value = fit
            if fit is None:
                self._initial_scene_deferred = True
                self._bootstrap_signature_hits = 0
                diagnostics = getattr(
                    self, "_last_template_fit_diagnostics", (0, 0, 0, 0),
                )
                problem_signature = (
                    "template-no-fit", base_level,
                    len(observations), len(partial_observations), diagnostics,
                    tuple(sorted(
                        (round(item.u, 2), round(item.v, 2),
                         round(item.width, 2), round(item.height, 2))
                        for item in (*observations, *partial_observations)
                    )),
                )
                if problem_signature != self._last_bootstrap_signature:
                    logger.debug(
                        "bootstrap por plantillas: ninguna configuracion valida "
                        "para %d completas y %d parciales; candidatos "
                        "una=%d dos=%d explicados_una=%d explicados_dos=%d; "
                        "se espera otro frame",
                        len(observations), len(partial_observations),
                        *diagnostics,
                    )
                self._last_bootstrap_signature = problem_signature
                return
            signature = self._template_bootstrap_signature(
                fit, observations, base_level,
            ) + (len(partial_observations),)
            if signature == self._last_bootstrap_signature:
                self._bootstrap_signature_hits += 1
            else:
                self._last_bootstrap_signature = signature
                self._bootstrap_signature_hits = 1
            has_upper = any(
                item.level == base_level + 1 for item in fit.assignments
            )
            # Tras deduplicar, un excedente de al menos dos observaciones
            # sobre los slots de N0 prueba dos niveles cuando A/B ya explican
            # de forma unica completas y parciales. Se reserva +1 porque una
            # sola caja inferior partida en dos fragmentos tambien puede
            # producir capacity+1 sin que exista N1.
            required_hits = (
                1
                if has_upper and evidence_count >= capacity + 2
                else self._cfg.confirmation.min_stable
            )
            logger.debug(
                "bootstrap por plantillas: fase=N0=%s hipotesis=%s ajuste=%.4f validacion=%d/%d",
                "A" if fit.phase == 0 else "B",
                "dos-niveles" if has_upper else "un-nivel", fit.mean_error,
                self._bootstrap_signature_hits, required_hits,
            )
            if self._bootstrap_signature_hits >= required_hits:
                self._apply_template_bootstrap_fit(fit, observations, base_level)
            return

        if not partials:
            self._bootstrap_solution_history.clear()
            self._last_bootstrap_partials.clear()
            self._last_bootstrap_complete_count = 0
            plain_signature = ("nivel-superior", base_level, len(occupied_rects), capacity)
            if plain_signature == self._last_bootstrap_signature:
                self._bootstrap_signature_hits += 1
            else:
                self._last_bootstrap_signature = plain_signature
                self._bootstrap_signature_hits = 1
            # Sin fragmentos parciales no hay ninguna oclusion 2D que resolver
            # -- ni si N0 esta vacio, ni si tiene pocas cajas y nada en N1.
            # Exigir capacidad completa aqui bloqueaba el ISO indefinidamente
            # durante la operacion normal (paleta que se llena una caja a la
            # vez, sin ambiguedad de composicion): la ausencia de recortes YA
            # es la prueba de que no hay nada que reconciliar.
            bootstrap_frames = self._cfg.confirmation.min_stable
            stable_top = self._bootstrap_signature_hits >= bootstrap_frames
            if stable_top:
                if len(occupied_rects) == capacity:
                    self._proven_full.add(base_level)
                self._finish_initial_bootstrap(
                    base_level,
                    "nivel superior estable sin mas recortes sin resolver",
                )
            return

        observed_partials = [(u, v, measured[0], measured[1]) for u, v, measured in partials]
        bootstrap_frames = self._cfg.confirmation.min_stable
        if not self._initial_scene_deferred:
            self._initial_scene_deferred = True
            logger.debug(
                "ESTADO INICIAL EN VALIDACION: se detecto una composicion N%d/N%d "
                "por reconciliar; el ISO 3D queda diferido hasta aplicar una "
                "solucion unica durante %d frames",
                base_level, base_level + 1, bootstrap_frames,
            )

        # Las identidades completas tambien tienen que seguir visibles. No se
        # valida un inventario inicial usando solo el estado persistido de un
        # frame anterior.
        if complete_count < len(occupied_rects):
            logger.debug(
                "bootstrap: esperando frame completo -- visibles_completas=%d, "
                "identidades_provisionales=%d; racha inicial reiniciada",
                complete_count, len(occupied_rects),
            )
            self._bootstrap_signature_hits = 0
            self._bootstrap_solution_history.clear()
            self._last_bootstrap_partials.clear()
            self._last_bootstrap_complete_count = 0
            return

        # Tras resolver una vez la combinatoria, los frames siguientes solo
        # tienen que demostrar que permanecen las mismas completas y los
        # mismos recortes. Reenumerar 100 hipotesis por frame hacia que el
        # arranque cayera a 0.6 fps sin aportar evidencia nueva.
        history_groups = self._bootstrap_solution_groups(self._bootstrap_solution_history)
        cached_solution = history_groups[0] if len(history_groups) == 1 else None
        aligned_partials = self._bootstrap_rects_aligned(
            self._last_bootstrap_partials, observed_partials,
        ) if self._last_bootstrap_partials else None
        if (
            cached_solution is not None
            and aligned_partials is not None
            and complete_count == self._last_bootstrap_complete_count
        ):
            self._bootstrap_signature_hits += 1
            self._bootstrap_solution_history.append(cached_solution)
            self._last_bootstrap_partials = aligned_partials
            logger.debug(
                "bootstrap: solucion unica %d/%d frames (validacion geometrica rapida); "
                "promueve_N%d=%s, reconstruidas_N%d=%d, ocultas_N%d=%d",
                self._bootstrap_signature_hits, bootstrap_frames, base_level + 1,
                sorted(cached_solution["promoted"]), base_level,
                len(cached_solution["partials"]), base_level,
                len(cached_solution["hidden"]),
            )
            if self._bootstrap_signature_hits >= bootstrap_frames:
                temporal = self._bootstrap_solution_groups(
                    self._bootstrap_solution_history,
                )[0]
                self._log_bootstrap_assignment(temporal, occupied_rects, "VALIDADA_FINAL")
                self._apply_bootstrap_solution(temporal)
                self._reconcile_initial_layers(parsed)
            return

        all_hypothesis_groups = [
            self._bootstrap_partial_hypotheses(u, v, measured, base_level, occupied_rects)
            for u, v, measured in partials
        ]
        # Un recorte sin ninguna hipotesis de encaje (tipico de una escalera
        # s(z) desviada para esta clase, o simple ruido) no puede tumbar TODA
        # la combinatoria: eso dejaba el bootstrap trabado para siempre en
        # cuanto aparecia un solo recorte irresoluble entre docenas validos.
        # Se excluye ese recorte de esta ronda (sigue detectandose el
        # siguiente frame, por si se resuelve solo) y se sigue intentando con
        # los que si encajan.
        hypothesis_groups = [group for group in all_hypothesis_groups if group]
        unfittable = len(all_hypothesis_groups) - len(hypothesis_groups)
        generated = 1
        for group in hypothesis_groups:
            generated *= len(group)
        quantum = max(self._cfg.partial_fit_tolerance, 0.01)
        raw_solutions: dict[tuple, dict] = {}
        hidden_candidate_cache: dict[
            tuple[int, ...], list[tuple[float, float, float, float]]
        ] = {}
        rejected = {"sin-encaje": unfittable, "solape": 0, "conteo": 0, "oculta": 0, "soporte": 0}
        # Antes, un solo recorte sin encaje abortaba TODO el frame antes de
        # llegar aca -- eso enmascaraba que, con todos los grupos no vacios,
        # el producto cartesiano de hipotesis por recorte puede explotar
        # (varias decenas de recortes con varias hipotesis cada uno son
        # millones de combinaciones). Sin este limite el proceso se cuelga
        # de verdad probando cada una; con el limite, este frame se trata
        # como "aun sin resolver" y se reintenta en el siguiente con menos
        # candidatos (la escena se estabiliza cuadro a cuadro).
        max_combinations = self._cfg.max_bootstrap_combinations
        if not hypothesis_groups or generated > max_combinations:
            if hypothesis_groups and generated > max_combinations:
                self._log.warning(
                    "bootstrap: combinatoria omitida -- %d combinaciones supera el "
                    "limite de seguridad (%d); se espera a que la escena reduzca la "
                    "ambiguedad en vez de enumerarlas todas",
                    generated, max_combinations,
                )
        else:
            for combination in itertools.product(*hypothesis_groups):
                reconstructed = [candidate[0] for candidate in combination]
                if any(
                    _rect_overlap_over_min(a, b) > self._cfg.max_same_level_overlap
                    for i, a in enumerate(reconstructed) for b in reconstructed[i + 1:]
                ):
                    rejected["solape"] += 1
                    continue
                promoted = set().union(*(set(candidate[1]) for candidate in combination))
                lower = [rect for cell, rect in occupied_rects.items() if cell not in promoted]
                lower.extend(reconstructed)
                hidden_count = capacity - len(lower)
                if hidden_count not in (0, 1):
                    rejected["conteo"] += 1
                    continue
                upper = [occupied_rects[cell] for cell in sorted(promoted)]
                promoted_key = tuple(sorted(promoted))
                if hidden_count == 0:
                    hidden_options = [()]
                else:
                    # La posicion de una caja totalmente oculta se deriva de
                    # quienes la OCULTAN (upper), no de cada variante de las
                    # reconstrucciones inferiores. Generarla una vez por
                    # topologia evita repetir cientos de rasterizaciones; el
                    # solape con `lower` si se valida para cada combinacion.
                    raw_hidden = hidden_candidate_cache.setdefault(
                        promoted_key,
                        self._bootstrap_hidden_candidates(
                            [], upper, consensus, base_level,
                        ),
                    )
                    hidden_options = [
                        (rect,) for rect in raw_hidden
                        if max(
                            (_rect_overlap_over_min(rect, other) for other in lower),
                            default=0.0,
                        ) <= self._cfg.max_same_level_overlap
                    ]
                if not hidden_options:
                    rejected["oculta"] += 1
                    continue
                for hidden in hidden_options:
                    all_lower = lower + list(hidden)
                    if not all(
                        self._bootstrap_top_is_supported(
                            rect, all_lower, base_level + 1,
                        )
                        for rect in upper
                    ):
                        rejected["soporte"] += 1
                        continue
                    signature = (
                        tuple(sorted(promoted)),
                        tuple(sorted(_quantize_position(r[0], r[1], quantum, r[2] >= r[3])
                                     for r in reconstructed)),
                        tuple(sorted(_quantize_position(r[0], r[1], quantum, r[2] >= r[3])
                                     for r in hidden)),
                    )
                    raw_solutions[signature] = {
                        "base_level": base_level,
                        "promoted": promoted,
                        "partials": reconstructed,
                        "hidden": list(hidden),
                        "partial_conflicts": [sorted(candidate[1]) for candidate in combination],
                    }

        solutions = self._bootstrap_solution_groups(list(raw_solutions.values()))

        problem_signature = (
            base_level,
            len(occupied_rects),
            tuple(sorted((round(u, 2), round(v, 2), round(m[0], 2), round(m[1], 2))
                         for u, v, m in partials)),
            tuple(sorted(
                (
                    tuple(sorted(solution["promoted"])),
                    tuple(sorted(_quantize_position(r[0], r[1], quantum, r[2] >= r[3])
                                 for r in solution["partials"])),
                    tuple(sorted(_quantize_position(r[0], r[1], quantum, r[2] >= r[3])
                                 for r in solution["hidden"])),
                )
                for solution in solutions
            )),
        )
        current_solution = solutions[0] if len(solutions) == 1 else None
        history_groups = self._bootstrap_solution_groups(self._bootstrap_solution_history)
        previous_solution = history_groups[0] if len(history_groups) == 1 else None
        same_problem = (
            current_solution is not None
            and previous_solution is not None
            and self._bootstrap_solutions_equivalent(previous_solution, current_solution)
        ) or (
            current_solution is None
            and problem_signature == self._last_bootstrap_signature
        )
        if same_problem:
            self._bootstrap_signature_hits += 1
        else:
            self._bootstrap_signature_hits = 1
            logger.debug(
                "bootstrap: completas_N%d=%d parciales=%d capacidad=%d combinaciones=%d "
                "factibles=%d descartes=%s",
                base_level, len(occupied_rects), len(partials), capacity,
                generated, len(solutions), rejected,
            )
            collapsed = len(raw_solutions) - len(solutions)
            if collapsed > 0:
                logger.debug(
                    "bootstrap: %d variantes numericas colapsadas en %d clase(s) "
                    "de equivalencia geometrica",
                    len(raw_solutions), len(solutions),
                )
            if len(solutions) == 1:
                self._log_bootstrap_assignment(
                    solutions[0], occupied_rects, "CANDIDATA_UNICA",
                )
            if len(solutions) > 1:
                logger.debug(
                    "bootstrap: AMBIGUO -- %d configuraciones esencialmente disjuntas; "
                    "no se muta el estado y se esperan mas observaciones",
                    len(solutions),
                )
                for number, candidate in enumerate(solutions[:5], start=1):
                    logger.debug(
                        "bootstrap: opcion %d promueve_N%d=%s reconstruye_N%d=%s oculta_N%d=%s",
                        number, base_level + 1,
                        sorted(candidate["promoted"]),
                        base_level,
                        [f"({r[0]:.3f},{r[1]:.3f},{r[2]:.3f}x{r[3]:.3f})"
                         for r in candidate["partials"]],
                        base_level,
                        [f"({r[0]:.3f},{r[1]:.3f},{r[2]:.3f}x{r[3]:.3f})"
                         for r in candidate["hidden"]],
                    )
                    self._log_bootstrap_assignment(
                        candidate, occupied_rects, f"OPCION_{number}",
                    )
                if len(solutions) > 5:
                    logger.debug("bootstrap: ... %d opciones adicionales omitidas", len(solutions) - 5)
        self._last_bootstrap_signature = problem_signature
        if current_solution is None:
            self._bootstrap_solution_history.clear()
            self._last_bootstrap_partials.clear()
            self._last_bootstrap_complete_count = 0
        elif same_problem:
            self._bootstrap_solution_history.append(current_solution)
            self._last_bootstrap_partials = observed_partials
            self._last_bootstrap_complete_count = complete_count
        else:
            self._bootstrap_solution_history = [current_solution]
            self._last_bootstrap_partials = observed_partials
            self._last_bootstrap_complete_count = complete_count
        if len(solutions) != 1:
            return
        needed = bootstrap_frames
        if self._bootstrap_signature_hits < needed:
            candidate = solutions[0]
            logger.debug(
                "bootstrap: solucion unica %d/%d frames; promueve_N%d=%s, "
                "reconstruidas_N%d=%d, ocultas_N%d=%d",
                self._bootstrap_signature_hits, needed, base_level + 1,
                sorted(candidate["promoted"]), base_level,
                len(candidate["partials"]), base_level,
                len(candidate["hidden"]),
            )
            return
        # La clasificacion y la geometria deben sobrevivir varios frames. La
        # solucion aplicada es la mediana temporal de esa misma clase fisica,
        # no las coordenadas accidentales del ultimo frame.
        temporal = self._bootstrap_solution_groups(self._bootstrap_solution_history)[0]
        self._log_bootstrap_assignment(temporal, occupied_rects, "VALIDADA_FINAL")
        self._apply_bootstrap_solution(temporal)
        self._reconcile_initial_layers(parsed)

    def _occlusion_pairs(
        self, detections: list[ParsedDetection],
    ) -> dict[int, int]:
        """Relaciona ``parcial -> completa`` usando solo geometria 2D.

        Esto todavia NO decide un nivel. El llamador debe anclar la parcial a
        una identidad confirmada y comprobar que el nivel inferior esta lleno.
        La confianza YOLO es deliberadamente irrelevante: una parcial puede
        tener mas confianza que la caja completa y no cambia su papel fisico.
        """
        partial_to_top: dict[int, int] = {}
        tau_overlap = self._cfg.tau_overlap
        tau_center = self._cfg.tau_overlap_center

        for i in range(len(detections)):
            bbox_i, _conf_i, _cls_i = detections[i]
            for j in range(i + 1, len(detections)):
                bbox_j, _conf_j, _cls_j = detections[j]
                if _intersection_over_min_area(bbox_i, bbox_j) < tau_overlap:
                    continue
                if _center_distance_over_min_side(bbox_i, bbox_j) > tau_center:
                    continue

                scale_i = _bbox_max_side(bbox_i)
                scale_j = _bbox_max_side(bbox_j)

                # Dos bboxes muy solapados y de TAMANO PARECIDO no son una
                # caja sobre otra: son dos detecciones de la MISMA caja. La
                # de un nivel superior esta mas cerca de la camara y se
                # proyecta netamente mas grande; si los tamanos casi
                # coinciden, no hay apilamiento que deducir y forzar uno
                # inventaba una caja fantasma en el nivel de arriba (que
                # ademas escapaba al chequeo de duplicados de _assign_cell,
                # porque ese solo compara celdas del mismo nivel).
                smaller, larger = sorted((scale_i, scale_j))
                if larger <= 0 or smaller / larger > self._cfg.max_duplicate_scale_ratio:
                    continue

                top_idx, low_idx = (i, j) if scale_i > scale_j else (j, i)
                partial_to_top[low_idx] = top_idx
        return partial_to_top
