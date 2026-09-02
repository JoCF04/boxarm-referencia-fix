from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm -- ver header completo en counter.py (paquete
# vision/palletizing/, historia acumulativa unica del modulo original).
# -----------------------------------------------------------------------
"""Mixin _FrameLoopMixin: puerta de entrada unica por frame (`update`),
gate de brazo/movimiento y el conteo (`_count_boxes`) propiamente dicho."""

import logging

from boxarm.history import record_pallet_completion

from .formulas import _bbox_center_and_size, _bbox_max_side, _project, _split_detection
from .templates.template_runtime import get_template_capacity
from .types import CellState, DetectionInput, FrameInput, FrameResult, GateState, GridDetection, LevelSource

logger = logging.getLogger(__name__)


class _FrameLoopMixin:
    """Puerta de entrada por frame: gate, conteo y cierre de ciclo de brazo."""

    # -- Puerta de entrada unica del cerebro ----------------------------------
    def update(self, frame: FrameInput) -> FrameResult:
        """Procesa un frame completo: decide si vale la pena mirarlo y, si
        vale, cuenta.

        Es el UNICO punto de entrada. El lazo de inferencia entrega lo que
        observo en crudo (bboxes, si vio el brazo, cuanto se movio el ROI) y
        recibe de vuelta que dibujar. No interpreta ninguno de esos numeros:
        el umbral de movimiento, el debounce del brazo y el cierre de ciclo
        son reglas de paletizado, y viven aca.
        """
        gate = self._advance_gate(frame)
        gate_changed = gate != self._gate
        if gate_changed:
            self._log_gate_change(gate, frame.motion_score)
        self._gate = gate

        if gate is not GateState.COUNTING:
            # `min_stable` exige frames COUNTING consecutivos. Una pausa por
            # robot/movimiento corta la racha; no se arrastra evidencia a
            # traves de una oclusion.
            self._pending_candidates.clear()
            return FrameResult([], gate, gate_changed, count_changed=False)

        # Recalcula la base antes de mirar detecciones: tambien cubre estados
        # restaurados o una ejecucion que ya tenia Ni completo al actualizar.
        self._advance_tracking_floor()
        before = self.total
        detections = self._count_boxes(frame.boxes)
        count_changed = self.total != before
        if count_changed:
            # Diagnostico completo cada vez que el conteo cambia -- no por
            # frame. Lo dispara el cerebro porque es el unico que sabe que
            # cambio; antes lo tenia que adivinar el lazo comparando totales.
            if self._initial_scene_deferred and not self._bootstrap_reconciled:
                logger.debug(
                    "ESTADO INICIAL PROVISIONAL: %d identidad(es) internas; "
                    "no se publican ni se dibujan hasta terminar la correccion",
                    len(self._occupied),
                )
            else:
                self._log_overlapping_cells()
                self._log_cell_table()
        return FrameResult(detections, gate, gate_changed, count_changed)

    def _advance_gate(self, frame: FrameInput) -> GateState:
        """Decide el estado del gate para este frame y cierra el ciclo de
        brazo cuando corresponde.

        El ciclo se cierra en el flanco "brazo visto -> brazo ausente de
        forma sostenida". Sostenida es la palabra clave: exigir
        `arm_debounce_frames` frames limpios evita que un unico frame en que
        el detector pierde el brazo se lea como un viaje terminado y habilite
        contar otra caja.
        """
        gate_cfg = self._cfg.gate

        if frame.arm_visible:
            self._frames_without_arm = 0
            self._arm_seen_in_cycle = True
        else:
            self._frames_without_arm += 1
            if (self._arm_seen_in_cycle
                    and self._frames_without_arm >= gate_cfg.arm_debounce_frames):
                self._arm_seen_in_cycle = False
                self._close_arm_cycle()

        if not gate_cfg.motion_pause_enabled:
            self._stable_frames = gate_cfg.motion_stable_frames
            return GateState.ARM_PAUSE if frame.arm_visible else GateState.COUNTING

        motion_detected = frame.motion_score > gate_cfg.motion_diff_threshold
        if frame.arm_visible or motion_detected:
            self._stable_frames = 0
        else:
            self._stable_frames = min(self._stable_frames + 1, gate_cfg.motion_stable_frames)

        if frame.arm_visible:
            return GateState.ARM_PAUSE
        if motion_detected:
            return GateState.MOTION_PAUSE
        if self._stable_frames < gate_cfg.motion_stable_frames:
            return GateState.SETTLING
        return GateState.COUNTING

    def _log_gate_change(self, gate: GateState, motion_score: float) -> None:
        gate_cfg = self._cfg.gate
        if gate is GateState.COUNTING:
            logger.debug("reanuda conteo: escena estable (%d/%d frames quietos, movimiento=%.2f)",
                        self._stable_frames, gate_cfg.motion_stable_frames, motion_score)
        elif gate is GateState.ARM_PAUSE:
            logger.debug("pausa conteo: brazo detectado (movimiento=%.2f)", motion_score)
        elif gate is GateState.MOTION_PAUSE:
            logger.debug("pausa conteo: movimiento en ROI %.2f > %.2f",
                        motion_score, gate_cfg.motion_diff_threshold)
        else:
            logger.debug("esperando estabilidad: %d/%d frames quietos (movimiento=%.2f)",
                        self._stable_frames, gate_cfg.motion_stable_frames, motion_score)

    def _count_boxes(self, detections: list[DetectionInput]) -> list[GridDetection]:
        """Procesa observaciones ``bbox+conf`` sin identidad temporal.

        La unica identidad persistente es espacial: ``(celda, nivel)``.
        """
        self._current_frame += 1
        results: list[GridDetection] = []
        parsed = self._deduplicate([_split_detection(det) for det in detections])

        # Vaciado real de la paleta: N frames COUNTING seguidos sin NINGUNA
        # deteccion, habiendo cajas ya confirmadas, es evidencia fisica de
        # que sacaron toda la carga -- no una oclusion momentanea (esa dura
        # menos que gate.empty_pallet_debounce_frames, calibrado al mismo
        # orden que motion_stable_frames). Reinicia TODO el estado de esta
        # paleta para la que va a entrar despues; ver reset_pallet(). Es la
        # UNICA transicion 1->0 de chi(g,z) de todo el paquete -- named y
        # auditable, no un decremento silencioso por frame.
        if parsed:
            self._empty_frames = 0
        else:
            self._empty_frames += 1
            if self._occupied and self._empty_frames >= self._cfg.gate.empty_pallet_debounce_frames:
                self._log.info(
                    "paleta vacia %d frames COUNTING seguidos -- se resetea el conteo "
                    "(%d caja(s) confirmada(s) se dan de baja)",
                    self._empty_frames, self.total,
                )
                # Vaciado real == la paleta se retiro ya cargada: es el
                # momento de dejar constancia en el historial, ANTES de que
                # reset_pallet() ponga total/box_class de nuevo en cero.
                record_pallet_completion(self._box_class, self.total, self._cam_tag)
                self.reset_pallet()

        # Short-circuit: paleta vacia y sin detecciones -> liberar clase y saltar logica
        if not parsed and not self._occupied:
            if self._box_class is not None:
                logger.debug("Paleta vacia y sin detecciones: se libera la denominacion %r", self._box_class)
                self._box_class = None
            return []

        # Filtrar clases extrañas y capturar la primera clase si la paleta estaba vacia
        filtered_parsed = []
        for det in parsed:
            bbox, _conf, cls_name = det
            if self._box_class is None and cls_name:
                self.set_box_class(cls_name)
                filtered_parsed.append(det)
            elif cls_name and cls_name != self._box_class:
                self._log.warning(
                    "ALERTA: Se detecto caja de clase %r pero la paleta esta fijada a %r. Ignorando.", 
                    cls_name, self._box_class
                )
            else:
                filtered_parsed.append(det)
        parsed = filtered_parsed

        preliminary_levels: list[int | None] = []
        preliminary_reasons: list[str] = []
        for bbox, _conf, cls_name in parsed:
            scale = _bbox_max_side(bbox)
            level, reason = self._assign_level(scale)
            if level is None and self._cfg.layout_mode == "auto":
                level, reason = self._nearest_level(scale), ""
            preliminary_levels.append(level)
            preliminary_reasons.append(reason)

        # Un bbox pequeno solapado no es otra caja: es la parte visible de
        # una identidad inferior. Solo puede validar un nivel superior si se
        # ancla a una caja YA confirmada y ese nivel inferior esta completo.
        # Sin ese ancla, una sola imagen 2D no permite demostrar la altura.
        partial_to_top = self._occlusion_pairs(parsed)

        # Emparejamiento 1 a 1 contra las celdas ya confirmadas, ANTES de
        # decidir niveles. Cada celda solo puede explicar una deteccion: si
        # hay tantas detecciones como celdas en una zona, no hay ninguna
        # caja nueva ahi por mas que se pisen. Sin esto, una deteccion que
        # era simplemente la re-observacion de una celda se interpretaba
        # como una caja apoyada encima (identica en la imagen: una caja
        # apilada tapa a la de abajo y se ve igual desde arriba).
        matched = self._match_to_cells(parsed)
        # Una deteccion adicional mas pequena puede ser solo el pedazo
        # visible de una identidad cuya observacion principal ya consumio
        # el match. Sirve como evidencia; nunca es candidata nueva.
        validated_partials = self._contained_validation_fragments(parsed, matched)
        occlusion_overrides: dict[int, int] = {}
        for partial_idx, top_idx in partial_to_top.items():
            partial_match = matched.get(partial_idx)
            top_match = matched.get(top_idx)

            # Si ambas observaciones ya tienen identidad en niveles
            # consecutivos, no hay ninguna caja nueva: la completa es REDET
            # de la superior y la parcial solo valida la inferior ocluida.
            if (partial_match is not None and top_match is not None
                    and top_match[1] == partial_match[1] + 1):
                validated_partials[partial_idx] = partial_match
                occlusion_overrides[partial_idx] = partial_match[1]
                occlusion_overrides[top_idx] = top_match[1]
                continue

            # Con una sola identidad confirmada, el matcher suele darsela al
            # bbox completo. La parcial demuestra que ESA identidad es la
            # inferior y libera la completa como candidata de i+1.
            lower_match = partial_match if partial_match is not None else top_match
            if lower_match is None:
                continue
            cell, lower_level = lower_match
            if lower_level + 1 >= self._cfg.levels:
                continue

            # Invariante fisico duro: una caja del nivel i+1 esta TRABADA
            # sobre el nivel i -- cae a caballo entre dos o mas cajas, nunca
            # alineada con una sola. Un par de oclusion demuestra que hay un
            # recorte y una caja completa, pero NO demuestra altura: sin este
            # chequeo la completa se promovia a i+1 en la posicion exacta de
            # la celda inferior, produciendo una torre de duplicados (misma
            # u, v y mismo tamano en los dos niveles) que es justamente la
            # re-deteccion que no se puede distinguir desde una cenital.
            top_bbox, _top_conf, _top_cls = parsed[top_idx]
            (tcx, tcy), (tw_px, th_px) = _bbox_center_and_size(top_bbox)
            top_footprint = self._measure_footprint(tcx, tcy, tw_px, th_px)
            top_u, top_v = _project(self._homography, tcx, tcy)
            if not self._has_interlocked_support(
                top_u, top_v, top_footprint, lower_level,
            ):
                assessment = self._support_polygon(
                    top_u, top_v, top_footprint, lower_level,
                )
                logger.debug(
                    "oclusion no promueve a nivel %d: la caja completa en (%.3f, %.3f) no esta "
                    "trabada sobre el nivel %d (contactos=%d, hull=%.0f%%, "
                    "centro_dentro=%s, degenerado=%s) -- se trata como re-deteccion",
                    lower_level + 1, top_u, top_v, lower_level,
                    assessment.contact_count,
                    assessment.hull_area_ratio * 100.0,
                    assessment.center_inside,
                    assessment.degenerate,
                )
                continue

            # Un recorte demuestra que ALGO lo tapa; que ese algo este trabado
            # encima demuestra que es una caja del nivel i+1; y una caja no se
            # apoya en el nivel i+1 mientras el nivel i tenga donde apoyar.
            # Encadenado: la existencia de la parcial PRUEBA que el nivel de
            # abajo esta lleno, sin necesidad de haber confirmado sus n cajas
            # una por una -- que es imposible cuando el video arranca con la
            # paleta ya apilada y algunas del piso nacieron tapadas.
            #
            # Solo durante el inventario inicial. Cerrado el primer ciclo, la
            # cuenta exacta ya es alcanzable porque cada caja nueva se observa
            # sin oclusion, y ahi manda la capacidad del template.
            if (
                (not self._arm_cycle_seen or self._template_baseline_pending)
                and lower_level not in self._proven_full
            ):
                self._proven_full.add(lower_level)
                logger.debug(
                    "nivel %d probado lleno por oclusion: la caja en (%.3f, %.3f) esta trabada "
                    "encima, y no se apila sobre un nivel con hueco -- no hace falta confirmar "
                    "sus cajas una por una",
                    lower_level, top_u, top_v,
                )
            if not self._level_is_full(lower_level):
                continue

            # El matcher voraz suele entregar la celda inferior al bbox
            # completo (por tener mas area). La evidencia parcial aclara la
            # identidad: el recorte re-observa la inferior y libera la
            # completa para evaluarla como caja de i+1.
            matched = {
                det_idx: key for det_idx, key in matched.items()
                if det_idx not in (partial_idx, top_idx) and key != lower_match
            }
            matched[partial_idx] = lower_match
            validated_partials[partial_idx] = lower_match
            occlusion_overrides[partial_idx] = lower_level
            occlusion_overrides[top_idx] = lower_level + 1

        stack_candidates: set[int] = set()
        candidate_boxes: dict[int, tuple[int, int, int, int]] = {}
        for idx, (bbox, _conf, cls_name) in enumerate(parsed):
            if idx in validated_partials:
                continue
            if idx not in matched:
                candidate_boxes[idx] = bbox
                continue
            (cx, cy), (w_px, h_px) = _bbox_center_and_size(bbox)
            measured = self._measure_footprint(cx, cy, w_px, h_px)
            u, v = _project(self._homography, cx, cy)
            if self._is_stack_candidate(idx, matched, u, v, measured):
                stack_candidates.add(idx)
                candidate_boxes[idx] = bbox
        stable_candidate_boxes = self._stable_candidate_boxes(candidate_boxes)
        for idx, (bbox, conf, cls_name) in enumerate(parsed):
            (cx, cy), (w_px, h_px) = _bbox_center_and_size(bbox)

            level, reason = preliminary_levels[idx], preliminary_reasons[idx]
            if idx in occlusion_overrides:
                level = occlusion_overrides[idx]
                reason = ""

            if level is None:
                results.append(GridDetection(bbox, None, None, CellState.REJECTED, reason, conf, box_class=cls_name))
                continue

            measured = self._measure_footprint(cx, cy, w_px, h_px)

            if idx in candidate_boxes and idx not in stable_candidate_boxes:
                results.append(GridDetection(
                    bbox, None, level, CellState.REJECTED,
                    "estabilizando", conf, box_class=cls_name
                ))
                continue

            # Ya emparejada con una celda confirmada: es esa celda, punto.
            # No se le busca nivel ni celda nueva -- ese camino es el que
            # convertia una simple re-observacion en una caja fantasma
            # apilada encima de si misma.
            #
            # EXCEPCION: una caja que el brazo acaba de apoyar ENCIMA de la
            # celda 7 tambien pisa el footprint de la celda 7, asi que el
            # matcher la reclama y la absorbe como re-deteccion antes de que
            # _stacking_level pueda promoverla. Eran indistinguibles aca, y
            # por eso el nivel 1 nunca aparecia.
            # Lo que las separa es el proceso: solo tras un ciclo de brazo,
            # con el nivel inferior lleno y soporte trabado, puede tratarse
            # como caja nueva encima. No existe identidad temporal.
            if idx in stack_candidates:
                logger.debug(
                    "apilamiento: deteccion pisa la celda=%d nivel=%d confirmada antes "
                    "-- no es re-deteccion, se evalua como caja encima",
                    matched[idx][0], matched[idx][1],
                )
            elif idx in validated_partials:
                cell, level = validated_partials[idx]
                results.append(GridDetection(
                    bbox, cell, level, CellState.VALIDATION,
                    "valida-oclusion-superior", conf, LevelSource.MATCH, box_class=cls_name
                ))
                continue

            elif idx in matched:
                cell, level = matched[idx]
                results.append(GridDetection(bbox, cell, level, CellState.REDET,
                                             confidence=conf,
                                             level_source=LevelSource.MATCH, box_class=cls_name))
                continue

            # Una identidad nueva persiste la mediana temporal de toda la
            # racha confirmada. El bbox actual queda solo para el overlay.
            geometry_bbox = stable_candidate_boxes.get(idx, bbox)
            (cx, cy), (gw_px, gh_px) = _bbox_center_and_size(geometry_bbox)
            measured = self._measure_footprint(cx, cy, gw_px, gh_px)

            # Puerta UNICA de decision de nivel. Antes esto eran dos bloques
            # sueltos (correccion por apilamiento y filtro de gravedad) con
            # sus `if idx not in occlusion_overrides` repartidos: nadie podia
            # decir, leyendo el codigo, cual mecanismo habia ganado. Ahora hay
            # un solo lugar que decide y que ademas dice QUIEN decidio.
            decision = self._resolve_level(
                cx, cy, measured, level, idx in occlusion_overrides)
            level, level_source = decision.level, decision.source
            if decision.reason:
                reason = decision.reason

            # Invariante monotona de una caja YA confirmada: si el candidato
            # estaba emparejado con el nivel i, una resolucion ambigua no
            # puede degradarlo a i, i-1, ...
            # Solo se permite separarlo como caja nueva cuando la evidencia
            # geometrica lo promueve ESTRICTAMENTE por encima de i.
            if idx in matched and level <= matched[idx][1]:
                cell, level = matched[idx]
                results.append(GridDetection(
                    bbox, cell, level, CellState.REDET,
                    confidence=conf,
                    level_source=LevelSource.MATCH, box_class=cls_name
                ))
                continue

            candidate_u, candidate_v = _project(self._homography, cx, cy)
            # Un recorte pertenece al nivel i-1 por definicion: algo lo tapa,
            # y ese algo esta encima. En vez de descartarlo, se reconstruye la
            # caja completa a la que pertenece creciendo hacia el lado que no
            # tiene vecina. Solo si esa reconstruccion es inequivoca.
            completed_partial = False
            if self._is_partial_footprint(measured, level):
                if (
                    not self._bootstrap_reconciled
                    and (not self._arm_cycle_seen or self._template_baseline_pending)
                ):
                    # En el inventario inicial un recorte es EVIDENCIA para
                    # la reconciliacion conjunta, no una identidad que pueda
                    # entrar a _occupied por orden de confianza YOLO. Si se
                    # completa aqui antes de procesar una caja entera vecina,
                    # puede ocupar su espacio y destruir la unica solucion
                    # fisica del bootstrap. `_reconcile_initial_layers`
                    # recibe igualmente `parsed` completo al final del frame.
                    results.append(GridDetection(
                        bbox, None, level, CellState.REJECTED,
                        "evidencia-bootstrap", conf, level_source,
                        box_class=cls_name,
                    ))
                    continue
                completion = self._complete_partial_footprint(
                    candidate_u, candidate_v, measured, level,
                )
                if completion is None:
                    consensus = self._level_footprint.get(level)
                    logger.debug(
                        "recorte NO reconstruible en nivel %d: fragmento %.3fx%.3f en "
                        "(%.3f, %.3f) contra caja canonica %s -- ninguna colocacion queda "
                        "bajo %.0f%% de solape con las vecinas, o hay mas de una posible",
                        level, measured[0], measured[1], candidate_u, candidate_v,
                        "desconocida" if consensus is None
                        else "%.3fx%.3f" % consensus,
                        getattr(self._cfg, "max_same_level_overlap", 0.10) * 100.0,
                    )
                    results.append(GridDetection(
                        bbox, None, level, CellState.REJECTED,
                        "recorte", conf, level_source, box_class=cls_name
                    ))
                    continue
                (candidate_u, candidate_v), measured = completion
                completed_partial = True
                cx, cy = self._unproject(candidate_u, candidate_v)
                logger.debug(
                    "recorte completado en nivel %d: fragmento -> caja %.3fx%.3f centrada en "
                    "(%.3f, %.3f), creciendo hacia el lado sin vecina",
                    level, measured[0], measured[1], candidate_u, candidate_v,
                )
            same_level_overlap = self._max_same_level_overlap(candidate_u, candidate_v, measured, level)
            overlap_limit = getattr(self._cfg, "max_same_level_overlap", 0.10)
            if same_level_overlap > overlap_limit:
                logger.debug(
                    "caja nueva RECHAZADA en nivel %d: pos=(%.3f, %.3f) tam=%.3fx%.3f pisa %.0f%% "
                    "a una vecina ya confirmada (limite %.0f%%) -- si esto se repite frame tras "
                    "frame, la caja real puede estar mal ubicada por error de deteccion/homografia, "
                    "o el robot la aposento demasiado cerca de la vecina",
                    level, candidate_u, candidate_v, measured[0], measured[1],
                    same_level_overlap * 100.0, overlap_limit * 100.0,
                )
                results.append(GridDetection(
                    bbox, None, level, CellState.REJECTED,
                    "solape-intranivel", conf, level_source, box_class=cls_name
                ))
                continue

            # Si el proceso ya demostro que es una caja NUEVA trabada sobre
            # el nivel inferior, no puede heredar una celda ocupada del nivel
            # superior solo porque un footprint inicial inflado la solape.
            # Una redeteccion superior real ya fue resuelta por `_matched`.
            # Regla de paletizado en su forma CONSTRUCTIVA: si el nivel
            # resuelto ya esta lleno, esta caja no puede estar ahi -- esta en
            # el de arriba. Antes se rechazaba como "nivel-lleno", que perdia
            # una caja real: que el nivel i-1 este lleno es justamente la
            # prueba de que todo lo nuevo pertenece a i.
            #
            # Solo llegan aca candidatas NUEVAS: una re-observacion de una
            # celda confirmada ya salio como REDET en el matcher, y una
            # parcial como VALIDATION. Asi que subir de nivel aqui no puede
            # convertir una re-deteccion en una caja fantasma.
            # Un recorte completado NO sube: que este tapado prueba que
            # pertenece al nivel de abajo, y es una de las cajas que ese nivel
            # ya contaba como suyas -- no una caja de mas.
            while (not completed_partial
                   and level + 1 < self._cfg.levels
                   and self._level_is_full(level)
                   and self._has_interlocked_support(
                       candidate_u, candidate_v, measured, level,
                   )):
                logger.debug(
                    "nivel %d lleno: la caja nueva en (%.3f, %.3f) no cabe ahi -- pasa al nivel %d",
                    level, candidate_u, candidate_v, level + 1,
                )
                level += 1
                level_source = LevelSource.STACKING

            force_new_stack_cell = (
                idx in stack_candidates and level_source is LevelSource.STACKING
            )
            cell, reason = self._assign_cell(
                cx, cy, level, measured, (w_px, h_px),
                reuse_occupied=not force_new_stack_cell,
            )
            if cell is None:
                results.append(GridDetection(bbox, None, level, CellState.REJECTED, reason,
                                             conf, level_source, box_class=cls_name))
                continue

            key = (cell, level)
            if key in self._occupied:
                # Estado confirmado: tamano y nivel son inmutables. La unica
                # correccion de posicion permitida ocurre en el camino de
                # matching, tras varias redetecciones completas y no ambiguas.
                results.append(GridDetection(bbox, cell, level, CellState.REDET, confidence=conf,
                                             level_source=level_source, box_class=cls_name))
                continue

            capacity = get_template_capacity(self._box_class or "")
            occupied_here = sum(1 for _g, z in self._occupied if z == level)
            if capacity is not None and occupied_here >= capacity:
                results.append(GridDetection(
                    bbox, None, level, CellState.REJECTED,
                    "nivel-lleno", conf, level_source, box_class=cls_name
                ))
                continue

            # Cada viaje cerrado concede UN credito. Una deteccion omitida no
            # destruye ese permiso: queda acumulado para recuperar backlog en
            # otro frame/ciclo. Sin credito, una identidad nueva posterior al
            # bootstrap sigue siendo fisicamente imposible.
            if (
                self._arm_cycle_seen
                and not self._template_baseline_pending
                and self._placement_credits <= 0
            ):
                # La posicion descubierta queda sin confirmar; no representa
                # una caja hasta que otro ciclo fisico permita aceptarla.
                if key not in self._rejected_in_cycle:
                    self._rejected_in_cycle.add(key)
                    self._log.warning(
                        "descartada celda=%d nivel=%d: no hay credito de colocacion "
                        "(un ciclo cerrado concede una caja) -- probable falso positivo",
                        cell, level,
                    )
                results.append(GridDetection(bbox, None, level, CellState.REJECTED,
                                             "ciclo-brazo", conf, level_source, box_class=cls_name))
                continue

            # Transicion 0 -> 1: la unica direccion permitida (Obs. 8.2 / Prop. 9.1)
            # El id y la posicion se vuelven permanentes SOLO en esta
            # transicion. Una candidata rechazada no consume ids ni deja
            # geometria fantasma en el estado confirmado.
            self._next_cell_by_level[level] = max(
                self._next_cell_by_level.get(level, 0), cell + 1,
            )
            self._dynamic_positions[key] = (candidate_u, candidate_v)
            self._occupied.add(key)
            self._cell_frame[key] = self._current_frame
            self._observe_footprint(key, measured)
            self._recompute_level_footprint(level)
            self.total += 1
            # Antes del primer ciclo de brazo la caja ya estaba ahi; despues,
            # la puso el brazo mientras mirabamos.
            if self._arm_cycle_seen and not self._template_baseline_pending:
                self._placement_credits -= 1
                self.placed += 1
            else:
                self.initial += 1
            self._log.info(
                "caja contada: celda=%d nivel=%d (decidido por %s) confianza=%s "
                "total=%d (inicial=%d colocadas=%d)",
                cell,
                level,
                level_source.value,
                f"{conf:.2f}" if conf is not None else "n/a",
                self.total, self.initial, self.placed,
            )
            self._advance_tracking_floor()
            results.append(GridDetection(bbox, cell, level, CellState.NEW, confidence=conf,
                                         level_source=level_source))

        # Antes del primer ciclo, una paleta que ya vino con varios niveles se
        # resuelve frontera por frontera Ni->N(i+1). Solo soluciones geometricas
        # unicas y repetidas pueden reasignar niveles e inferir cajas ocultas.
        # El solver se congela al cerrar el ultimo nivel observado.
        self._reconcile_initial_layers(parsed)

        # Cajas distintas vistas en el frame, NO detecciones crudas: el
        # detector emite a veces dos bboxes de la misma caja y el HUD
        # mostraba "Visibles: 16" con 15 cajas reales en pantalla.
        self.visible = len(parsed)
        return results

    def _close_arm_cycle(self) -> None:
        """Marca que el brazo termino un ciclo (entro, apoyo y salio).

        Lo dispara `_advance_gate` al confirmar la transicion brazo
        presente -> ausente sostenida durante `gate.arm_debounce_frames`.
        Cada cierre concede un credito. Si vision no confirma la caja durante
        ese ciclo, el credito se conserva y permite recuperar el atraso en un
        frame posterior. Una identidad nueva sin credito sigue rechazada."""
        self._arm_cycle_seen = True
        self._placement_credits += 1
        logger.debug(
            "ciclo de brazo cerrado: credito concedido; pendientes=%d",
            self._placement_credits,
        )
        self._rejected_in_cycle.clear()
        self._logged_candidate_diagnostics.clear()
