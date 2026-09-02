from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm -- ver header completo en counter.py (paquete
# vision/palletizing/, historia acumulativa unica del modulo original).
# -----------------------------------------------------------------------
"""Mixin _MatchingMixin: confirmacion temporal de candidatas nuevas,
deduplicacion de detecciones repetidas y emparejamiento 1 a 1 contra
celdas ya confirmadas."""

import logging

from .formulas import (
    _bbox_center_and_size,
    _bbox_iou,
    _bbox_max_side,
    _footprint_overlap_over_min,
    _footprint_containment,
    _intersection_over_min_area,
    _observed_median,
    _project,
)
from .types import ParsedDetection

logger = logging.getLogger(__name__)


class _MatchingMixin:
    """Confirmacion temporal, deduplicacion y emparejamiento a celdas."""

    def _stable_candidate_boxes(
        self, candidates: dict[int, tuple[int, int, int, int]]
    ) -> dict[int, tuple[int, int, int, int]]:
        """Confirma observaciones nuevas durante N frames consecutivos.

        La asociación por IoU solo existe mientras la candidata está
        pendiente. Nunca identifica ni actualiza una caja ya confirmada. La
        geometria persistida es la mediana temporal, no el ultimo bbox.
        """
        if not candidates:
            self._pending_candidates.clear()
            return {}
        required = self._cfg.confirmation.min_stable
        if required <= 1:
            self._pending_candidates.clear()
            return dict(candidates)

        pairs = []
        for idx, bbox in candidates.items():
            for pending_idx, history in enumerate(self._pending_candidates):
                iou = _bbox_iou(bbox, history[-1])
                if iou >= self._cfg.confirmation.same_box_iou:
                    pairs.append((iou, idx, pending_idx))
        pairs.sort(reverse=True)

        matched: dict[int, int] = {}
        used_pending: set[int] = set()
        for _iou, idx, pending_idx in pairs:
            if idx in matched or pending_idx in used_pending:
                continue
            matched[idx] = pending_idx
            used_pending.add(pending_idx)

        stable: dict[int, tuple[int, int, int, int]] = {}
        next_pending: list[list[tuple[int, int, int, int]]] = []
        for idx, bbox in candidates.items():
            pending_idx = matched.get(idx)
            history = [] if pending_idx is None else list(self._pending_candidates[pending_idx])
            history.append(bbox)
            if len(history) >= required:
                recent = history[-required:]
                stable[idx] = tuple(
                    int(_observed_median(box[axis] for box in recent))
                    for axis in range(4)
                )
            else:
                next_pending.append(history)
        self._pending_candidates = next_pending
        return stable

    def _deduplicate(self, parsed: list[ParsedDetection]) -> list[ParsedDetection]:
        """Colapsa las detecciones que son la MISMA caja vista dos veces.

        El detector emite a veces dos bboxes casi identicos para una sola
        caja. Hay que resolverlo ACA, sobre la lista de detecciones cruda,
        antes de cualquier otra decision: si se deja pasar, mas adelante
        cada copia se compara por separado contra las cells y la segunda
        termina interpretada como una caja supported encima de la primera --
        una caja fantasma en el nivel de arriba, que ademas sobrecuenta.

        El criterio es overlap alto Y tamano parecido: dos bboxes del mismo
        tamano encimados no pueden ser una caja sobre otra (de serlo, la de
        abajo saldria recortada o no saldria). Si los tamanos difieren
        bastante, se dejan las dos: ahi si puede haber apilamiento u
        oclusion real, que resuelve _occlusion_pairs.
        """
        alive: list[ParsedDetection] = []
        for det in parsed:
            bbox, conf, cls_name = det
            duplicate_of = None
            for idx, (bbox_v, conf_v, cls_name_v) in enumerate(alive):
                if _intersection_over_min_area(bbox, bbox_v) < self._cfg.tau_cell_overlap:
                    continue
                scale = _bbox_max_side(bbox)
                scale_other = _bbox_max_side(bbox_v)
                smaller, larger = sorted((scale, scale_other))
                if larger > 0 and smaller / larger > self._cfg.max_duplicate_scale_ratio:
                    duplicate_of = idx
                    break
            if duplicate_of is None:
                alive.append(det)
                continue
            # Se conserva la de mayor confianza: es la mejor observacion de
            # esa caja, y su bbox suele estar menos recortado.
            if (conf or 0.0) > (alive[duplicate_of][1] or 0.0):
                alive[duplicate_of] = det
        if len(alive) != len(parsed):
            logger.debug("deduplicacion: %d detecciones -> %d cajas distintas", len(parsed), len(alive))
        return alive

    def _match_to_cells(self, parsed: list[ParsedDetection]) -> dict[int, tuple[int, int]]:
        """Asigna a cada deteccion, como mucho, UNA celda ya confirmada.

        Emparejamiento voraz por overlap de footprint, uno a uno: una celda
        no puede explicar dos detecciones ni al reves. Es la pieza que
        faltaba -- una caja apilada tapa a la de abajo, asi que desde una
        camara cenital "celda 7 re-detectada" y "caja nueva encima de la
        celda 7" dan la MISMA imagen. Lo unico que las separa es contar:
        si hay una sola deteccion sobre una sola celda, es esa celda; solo
        una deteccion que sobra, sin celda libre que la explique, puede ser
        una caja nueva.
        """
        candidates = []
        for idx, (bbox, _conf, _cls_name) in enumerate(parsed):
            (cx, cy), (w_px, h_px) = _bbox_center_and_size(bbox)
            u, v = _project(self._homography, cx, cy)
            fp_det = self._measure_footprint(cx, cy, w_px, h_px)
            for (g, z) in self._footprint:
                if z < self._tracking_floor_level or (g, z) not in self._occupied:
                    continue
                fp = self._canonical_footprint((g, z))
                pos = self._dynamic_positions.get((g, z)) if self._cfg.layout_mode == "auto" else None
                if pos is None and self._cfg.layout_mode == "auto":
                    continue
                if pos is None:
                    pos = self._cell_position(g, z)
                overlap = _footprint_overlap_over_min((u, v), fp_det, pos, fp)
                containment = _footprint_containment((u, v), fp_det, pos, fp)
                min_containment = self._min_redetection_containment(fp_det, z)
                if (
                    overlap >= self._cfg.tau_cell_overlap
                    and containment >= min_containment
                ):
                    candidates.append((containment, overlap, fp_det[0] * fp_det[1], idx, (g, z)))

        # Si varias cajas apiladas comparten XY, una deteccion puede solapar
        # por igual las celdas de
        # nivel 0, 1, 2... La visible es la superior: ante el mismo solape se
        # conserva el nivel confirmado MAS ALTO. Sin este desempate ganaba la
        # celda insertada primero, normalmente nivel 0, y la caja "caia".
        # Si una identidad produce una observacion completa y otro recorte
        # contenido, la completa debe conservar el match. De lo contrario el
        # recorte (que tambien puede tener contencion=1) consumia la celda y
        # la observacion completa quedaba como una falsa candidata nueva.
        candidates.sort(key=lambda c: (-c[4][1], -c[0], -c[1], -c[2]))
        matched: dict[int, tuple[int, int]] = {}
        used_cells: set[tuple[int, int]] = set()
        for _containment, _overlap, _area, idx, celda in candidates:
            if idx in matched or celda in used_cells:
                continue
            matched[idx] = celda
            used_cells.add(celda)
        return matched

    def _contained_validation_fragments(
        self,
        parsed: list[ParsedDetection],
        matched: dict[int, tuple[int, int]],
    ) -> dict[int, tuple[int, int]]:
        """Devuelve recortes sobrantes de identidades ya observadas.

        La relacion es deliberadamente uno-a-muchos solo para VALIDATION:
        la deteccion principal conserva la identidad y un bbox adicional
        puede validar que queda visible un pedazo inferior. El recorte debe
        estar contenido y ser estrictamente menor; puede achicarse, pero
        nunca crecer ni crear otra caja.
        """
        validations: dict[int, tuple[int, int]] = {}
        used_cells = set(matched.values())
        tolerance = max(
            getattr(self._cfg, "partial_fit_tolerance", 0.0),
            1.0 / max(getattr(self._cfg, "occupancy_grid", 100), 1),
        )

        for idx, (bbox, _conf, _cls_name) in enumerate(parsed):
            if idx in matched:
                continue
            (cx, cy), (w_px, h_px) = _bbox_center_and_size(bbox)
            u, v = _project(self._homography, cx, cy)
            fp_det = self._measure_footprint(cx, cy, w_px, h_px)
            candidates: list[tuple[float, tuple[int, int]]] = []
            for key in used_cells:
                g, z = key
                fp = self._canonical_footprint(key)
                pos = self._dynamic_positions.get(key) if self._cfg.layout_mode == "auto" else None
                if pos is None and self._cfg.layout_mode == "auto":
                    continue
                if pos is None:
                    pos = self._cell_position(g, z)

                # Invariante fisico: un recorte solo pierde extension. Una
                # deteccion mayor en cualquier eje no puede reinterpretarse
                # como el pedazo visible de esta identidad.
                if fp_det[0] > fp[0] + tolerance or fp_det[1] > fp[1] + tolerance:
                    continue
                area_ratio = (fp_det[0] * fp_det[1]) / max(fp[0] * fp[1], 1e-9)
                if area_ratio >= self._cfg.max_duplicate_scale_ratio:
                    continue
                containment = _footprint_containment((u, v), fp_det, pos, fp)
                if containment < self._min_redetection_containment(fp_det, z):
                    continue
                candidates.append((containment, key))

            if candidates:
                validations[idx] = max(candidates, key=lambda item: (item[0], item[1][1]))[1]
        return validations
