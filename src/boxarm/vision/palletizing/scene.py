from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm -- ver header completo en counter.py (paquete
# vision/palletizing/, historia acumulativa unica del modulo original).
# -----------------------------------------------------------------------
"""Mixin _SceneMixin: escena resuelta para el renderizador isometrico,
diagnostico de overlaps intranivel y verificacion de cierre de paleta."""

import logging
from dataclasses import replace

import numpy as np

from .formulas import (
    _bbox_center_and_size,
    _bbox_max_side,
    _footprint_overlap_over_min,
    _observed_median,
    _project,
    _scene_overlap_corners,
    _split_detection,
)
from .types import DetectionInput, SceneBox, SceneOverlap, SceneState

logger = logging.getLogger(__name__)


class _SceneMixin:
    """Escena resuelta para el ISO, diagnostico y verificacion de cierre."""

    def provisional_boxes(
        self,
        detections: list[DetectionInput],
        height_ratio: float,
        level_tops: list[float],
    ) -> list[SceneBox]:
        """Proyecta observaciones todavia no confirmadas sin tocar ``chi``.

        Esta salida es exclusivamente visual: no llama a la confirmacion
        temporal, no asigna celdas y no modifica ``total``. Una deteccion
        que ya explica una identidad confirmada se omite. La unica excepcion
        es una posible caja apilada despues (o durante el cierre) de un ciclo
        de brazo, siempre que la geometria tambien la sostenga.
        """
        parsed = self._deduplicate([_split_detection(det) for det in detections])
        active_class = self._box_class or next(
            (cls_name for _bbox, _confidence, cls_name in parsed if cls_name),
            None,
        )
        parsed = [
            det for det in parsed
            if not det[2] or active_class is None or det[2] == active_class
        ]
        if not parsed:
            return []

        bootstrap_pending = (
            self._cfg.layout_mode == "auto"
            and self._initial_scene_deferred
            and not self._bootstrap_reconciled
        )
        matched = self._match_to_cells(parsed)
        validation_fragments = (
            set()
            if bootstrap_pending
            else self._contained_validation_fragments(parsed, matched)
        )
        preview: list[SceneBox] = []

        for idx, (bbox, _confidence, cls_name) in enumerate(parsed):
            if idx in validation_fragments:
                continue

            (cx, cy), (w_px, h_px) = _bbox_center_and_size(bbox)
            measured = self._measure_footprint(cx, cy, w_px, h_px)
            u, v = _project(self._homography, cx, cy)
            matched_key = matched.get(idx)

            if matched_key is not None:
                _cell, matched_level = matched_key
                if bootstrap_pending:
                    level = matched_level
                else:
                    # Un fragmento que coincide con una caja inferior no
                    # puede convertirse visualmente en una caja nueva del
                    # nivel superior. La parcial sirve para validar
                    # oclusion, pero no es evidencia completa de apilamiento.
                    if self._is_partial_footprint(measured, matched_level):
                        continue
                    stack_candidate = self._is_stack_candidate(idx, matched, u, v, measured)
                    # Mientras el debounce termina de cerrar el viaje del brazo,
                    # todavia no hay credito de colocacion. Para la PREVIA basta
                    # evidencia fisica: ciclo abierto, nivel lleno y soporte
                    # trabado. La confirmacion real conserva sus puertas duras.
                    pending_stack = (
                        self._arm_seen_in_cycle
                        and matched_level + 1 < self._cfg.levels
                        and self._level_is_full(matched_level)
                        and self._has_interlocked_support(u, v, measured, matched_level)
                    )
                    if not stack_candidate and not pending_stack:
                        continue
                    level = matched_level + 1
            else:
                scale = _bbox_max_side(bbox)
                level, _reason = self._assign_level(scale)
                if level is None:
                    if self._cfg.layout_mode != "auto":
                        continue
                    level = self._nearest_level(scale)

                if self._cfg.layout_mode == "auto":
                    # Variante visual y sin efectos laterales de
                    # _stacking_level: reutiliza exactamente sus pruebas de
                    # soporte/llenado, pero no escribe caches de diagnostico
                    # ni emite logs dos veces por frame.
                    level = self._tracking_floor_level
                    for support_level in range(
                        self._cfg.levels - 1,
                        self._tracking_floor_level - 1,
                        -1,
                    ):
                        if not self._has_interlocked_support(
                            u, v, measured, support_level,
                        ):
                            continue
                        level = support_level
                        if (
                            self._level_is_full(support_level)
                            and support_level + 1 < self._cfg.levels
                        ):
                            level = support_level + 1
                        break

            if level < 0 or level >= self._cfg.levels:
                continue

            # Una observacion sin celda conocida tampoco puede inaugurar un
            # nivel si su footprint ya demuestra que es un recorte. Las
            # parciales se conservan como evidencia interna/validacion, no
            # como cajas nuevas visibles del ISO.
            if level > self._tracking_floor_level:
                partial = self._is_partial_footprint(measured, level)
                # Si aun no hay tres muestras del nivel nuevo, usa el
                # footprint del nivel conocido mas cercano y la razon de
                # escalas. Asi un recorte aislado no inaugura N2/N3 solo
                # porque su lado mayor cae cerca de ese peldaño.
                if not partial and matched_key is None:
                    references = [z for z in self._level_footprint if z < level]
                    if references:
                        reference = max(references)
                        scale_ratio = self._ladder[level] / self._ladder[reference]
                        expected = tuple(
                            side * scale_ratio for side in self._level_footprint[reference]
                        )
                        ratio = getattr(self._cfg, "min_complete_side_ratio", 0.70)
                        partial = (
                            max(measured) < max(expected) * ratio
                            or min(measured) < min(expected) * ratio
                        )
                if partial:
                    continue

            height = height_ratio * min(measured)
            if level < len(level_tops):
                z0 = level_tops[level]
            else:
                base = level_tops[-1] if level_tops else 0.0
                missing_levels = level - max(len(level_tops) - 1, 0)
                z0 = base + max(missing_levels, 0) * height

            preview.append(SceneBox(
                cell=-1,
                level=level,
                u=u,
                v=v,
                z0=z0,
                side_a=measured[0],
                side_b=measured[1],
                height=height,
                box_class=cls_name or self._box_class or "",
                status="initializing" if bootstrap_pending else "confirming",
            ))

        return preview

    def _log_cell_table(self) -> None:
        """Vuelca todas las celdas confirmadas con su posicion, su tamano y
        cuanto las sostiene el nivel de abajo.

        Es el estado completo del contador en una tabla: si una caja aparece
        dibujada donde no va, aca se ve si el problema es su posicion (mal
        emparejada) o su tamano (footprint medido bajo oclusion). La columna
        `apoyo` dice que fraccion de esa caja esta sostenida por el nivel
        inferior -- para una caja de nivel 0 no aplica, y para una apilada
        deberia ser alta aunque este trabada entre dos de abajo.
        """
        if not self._occupied:
            logger.debug("celdas: ninguna confirmada todavia")
            return
        logger.debug("celdas confirmadas (%d):", len(self._occupied))
        for g, z in sorted(self._occupied, key=lambda k: (k[1], k[0])):
            raw_footprint = self._footprint.get((g, z))
            try:
                u, v = self._cell_position(g, z)
            except KeyError:
                self._log.warning("  celda=%-3d nivel=%d  SIN POSICION REGISTRADA", g, z)
                continue
            if raw_footprint is None:
                self._log.warning("  celda=%-3d nivel=%d  pos=(%.3f, %.3f)  SIN FOOTPRINT", g, z, u, v)
                continue

            footprint = self._canonical_footprint((g, z))
            support = "-" if z == 0 else f"{self._support_coverage(u, v, footprint, z - 1) * 100:.0f}%"
            raw_note = ""
            if not np.allclose(raw_footprint, footprint, atol=1e-6):
                raw_note = f" bbox_inicial={raw_footprint[0]:.3f}x{raw_footprint[1]:.3f}"
            logger.debug(
                "  celda=%-3d nivel=%d  pos=(%.3f, %.3f)  tam_canon=%.3fx%.3f%s  apoyo=%s",
                g, z, u, v, footprint[0], footprint[1], raw_note, support,
            )

    def _overlapping_cells(self, threshold: float | None = None) -> list[tuple[int, int, int, float]]:
        """Pares (celda_a, celda_b, nivel, solape) de celdas confirmadas del
        MISMO nivel cuyos footprints se pisan mas que `threshold`.

        Dos cajas de un nivel estan lado a lado sobre la paleta, asi que un
        overlap apreciable no es geometria: senala una celda duplicada (la
        misma caja contada dos veces con el centroide corrido) y por lo tanto
        un sobreconteo. Solo diagnostico -- no toca chi ni el total.

        `threshold` default None usa `cfg.overlap_warn_ratio`
        (configs/palletizing.yaml) -- NO es max_same_level_overlap: ese
        bloquea contar una caja NEW, este solo decide si se loguea/pinta
        una alerta de posible duplicado."""
        if threshold is None:
            threshold = self._cfg.overlap_warn_ratio
        pairs = []
        cells = sorted(self._occupied)
        for i, (g_a, z_a) in enumerate(cells):
            for g_b, z_b in cells[i + 1:]:
                if z_a != z_b:
                    continue
                if (g_a, z_a) not in self._footprint or (g_b, z_b) not in self._footprint:
                    continue
                fp_a = self._canonical_footprint((g_a, z_a))
                fp_b = self._canonical_footprint((g_b, z_b))
                # Diagnostico: nunca debe tumbar el hilo de inferencia por una
                # celda sin posicion registrada -- se omite ese par y listo.
                pos_a = self._dynamic_positions.get((g_a, z_a)) if self._cfg.layout_mode == "auto" else None
                pos_b = self._dynamic_positions.get((g_b, z_b)) if self._cfg.layout_mode == "auto" else None
                if self._cfg.layout_mode == "auto" and (pos_a is None or pos_b is None):
                    continue
                if self._cfg.layout_mode != "auto":
                    pos_a, pos_b = self._cell_position(g_a, z_a), self._cell_position(g_b, z_b)
                overlap = _footprint_overlap_over_min(pos_a, fp_a, pos_b, fp_b)
                if overlap > threshold:
                    pairs.append((g_a, g_b, z_a, overlap))
        return sorted(pairs, key=lambda p: -p[3])

    def _log_overlapping_cells(self, threshold: float | None = None) -> None:
        """Vuelca a log los solapes detectados por `overlapping_cells()`."""
        if threshold is None:
            threshold = self._cfg.overlap_warn_ratio
        pairs = self._overlapping_cells(threshold)
        if not pairs:
            logger.debug("overlaps: ninguno por encima de %.0f%% entre %d cells", threshold * 100.0, len(self._occupied))
            return
        self._log.warning("overlaps: %d pairs por encima de %.0f%% (posibles cells duplicadas)",
                       len(pairs), threshold * 100.0)
        for g_a, g_b, z, overlap in pairs:
            self._log.warning(
                "  celda %d y celda %d (nivel %d) se pisan %.0f%% -- %s en (%.3f, %.3f) y (%.3f, %.3f)",
                g_a, g_b, z, overlap * 100.0,
                "MISMA CAJA duplicada" if overlap >= self._cfg.tau_cell_overlap else "revisar",
                *self._cell_position(g_a, z), *self._cell_position(g_b, z),
            )

    # -- Escena resuelta para el renderizador ----------------------------------
    def scene_state(self, height_ratio: float) -> SceneState:
        """Devuelve la carga entera ya resuelta, lista para dibujar.

        Todo lo que antes calculaba el renderizador vive aca: unificar el
        footprint dentro de un nivel, derivar la altura de cada nivel y
        apilar las cotas. La razon es que son decisiones sobre la CARGA, no
        sobre como se ve: si el ISO las tomaba por su cuenta, habia dos
        fuentes de verdad sobre el tamano de una caja y podian discrepar.

        Sobre la unificacion de footprint: se hace DENTRO de cada nivel, no
        entre niveles. La homografia esta calibrada sobre el piso del pallet
        (z=0), asi que una caja mas alta esta mas cerca de la camara y se
        proyecta mas grande -- esa diferencia entre niveles es senal, no
        error, y borrarla dejaria toda la pila dibujada como una sola capa.
        Dentro de un nivel, en cambio, toda diferencia de tamano si es error
        de medicion, y para eso ya esta el consenso por nivel.

        `height_ratio` es la proporcion visual de extrusion
        (IsometricConfig.visual_height_ratio). NO es la altura fisica Z: esa
        no es observable desde un bbox de una sola camara cenital.
        """
        # Altura por nivel: se deriva del lado menor de sus cajas, asi que
        # un nivel sin cajas propias no tiene altura medible y hereda el
        # promedio de los que si la tienen.
        heights: dict[int, float] = {}
        for cell, level in self._occupied:
            consensus = self._level_footprint.get(level)
            own = self._footprint.get((cell, level))
            sides = consensus or own
            if sides is None:
                continue
            height = height_ratio * min(sides)
            heights[level] = min(heights[level], height) if level in heights else height

        fallback = _observed_median(heights.values()) if heights else height_ratio

        def level_height(level: int) -> float:
            return heights.get(level, fallback)

        # El dominio de calibracion puede admitir mas niveles, pero el render
        # solo extruye los observados; los futuros no agregan altura vacia.
        scene_levels = max((level for _cell, level in self._occupied), default=0) + 1
        level_tops = [0.0]
        for level in range(scene_levels):
            level_tops.append(level_tops[-1] + level_height(level))

        boxes: list[SceneBox] = []
        for cell, level in sorted(self._occupied, key=lambda k: (k[1], k[0])):
            own = self._footprint.get((cell, level))
            if own is None:
                continue  # no deberia pasar: solo se confirma con footprint medido
            try:
                u, v = self._cell_position(cell, level)
            except KeyError:
                self._log.warning("celda=%d nivel=%d sin posicion registrada -- se omite del render",
                               cell, level)
                continue
            consensus = self._level_footprint.get(level)
            if consensus is None:
                side_a, side_b = own
            else:
                # El consenso viene como (lado largo, lado corto); se aplica en
                # el orden propio de esta caja para que una girada 90 grados
                # se siga viendo girada.
                long_side, short_side = consensus
                side_a, side_b = (
                    (long_side, short_side) if own[0] >= own[1] else (short_side, long_side)
                )
            boxes.append(SceneBox(
                cell=cell, level=level, u=u, v=v,
                side_a=side_a, side_b=side_b,
                z0=level_tops[level], height=level_height(level),
                box_class=self._box_class or "",
            ))

        by_key = {(box.cell, box.level): box for box in boxes}
        overlaps: list[SceneOverlap] = []
        for cell_a, cell_b, level, ratio in self._overlapping_cells():
            box_a, box_b = by_key.get((cell_a, level)), by_key.get((cell_b, level))
            if box_a is None or box_b is None:
                continue
            corners = _scene_overlap_corners(
                box_a.u, box_a.v, box_a.side_a, box_a.side_b,
                box_b.u, box_b.v, box_b.side_a, box_b.side_b,
            )
            if corners is None:
                continue
            u0, v0, u1, v1 = corners
            overlaps.append(SceneOverlap(
                cell_a=cell_a, cell_b=cell_b, level=level, ratio=ratio,
                u0=u0, v0=v0, u1=u1, v1=v1,
                z0=box_a.z0, height=box_a.height,
            ))

        bootstrap_pending = (
            self._cfg.layout_mode == "auto"
            and self._initial_scene_deferred
            and not self._bootstrap_reconciled
        )
        if bootstrap_pending:
            # El inventario interno todavia puede cambiar durante la solucion
            # conjunta A/B. Se muestra de inmediato, pero explicitamente como
            # geometria provisional: no forma parte de ``boxes`` ni de los
            # totales confirmados que consumen persistencia y automatizacion.
            return SceneState(
                boxes=[],
                overlaps=[],
                level_tops=level_tops,
                total_height=level_tops[-1],
                total=0,
                initial=0,
                placed=0,
                levels=scene_levels,
                provisional_boxes=[
                    replace(box, status="initializing") for box in boxes
                ],
                validating_initial=True,
            )

        self._log_level_composition(boxes, level_tops)
        return SceneState(
            boxes=boxes, overlaps=overlaps,
            level_tops=level_tops, total_height=level_tops[-1],
            total=self.total, initial=self.initial, placed=self.placed,
            levels=scene_levels,
        )

    def _log_level_composition(self, boxes: list[SceneBox], level_tops: list[float]) -> None:
        """Vuelca cuantas cajas quedaron en cada nivel, con que tamano y a
        que altura arrancan -- solo cuando la composicion cambia.

        Es el contraste directo contra la realidad: si el ISO dibuja una
        pila que no existe, aca se ve si el problema es el reparto por
        niveles o el tamano de dibujo. Un nivel 1 cuyo tamano no es mayor
        que el del nivel 0 delata un footprint medido bajo oclusion."""
        per_level: dict[int, int] = {}
        for box in boxes:
            per_level[box.level] = per_level.get(box.level, 0) + 1
        signature = tuple(sorted(per_level.items()))
        if signature == self._last_level_signature:
            return
        self._last_level_signature = signature
        for level in sorted(per_level):
            consensus = self._level_footprint.get(level)
            logger.debug(
                "iso nivel %d: %d caja(s)  tam=%s  z0=%.3f",
                level, per_level[level],
                "n/d" if consensus is None else f"{consensus[0]:.3f}x{consensus[1]:.3f}",
                level_tops[level],
            )

    # -- Seccion 9: verificacion de cierre --------------------------------------
    def missing_cells(self) -> list[tuple[int, int]]:
        """Celdas (g, z) que el patron espera ocupadas y todavia no lo
        estan -- para el chequeo de cierre antes de embolsar."""
        if self._cfg.layout_mode == "auto":
            return []  # no hay un patron esperado contra el cual comparar
        missing = []
        for z, layout in enumerate(self._cfg.levels_layout):
            for g in range(len(layout.cells)):
                if (g, z) not in self._occupied:
                    missing.append((g, z))
        return missing

    def expected_total(self) -> int:
        if self._cfg.layout_mode == "auto":
            return self.total
        return sum(len(layout.cells) for layout in self._cfg.levels_layout)

    def is_complete(self) -> bool:
        if self._cfg.layout_mode == "auto":
            return False  # el total esperado no se conoce de antemano
        return self.total >= self.expected_total()
