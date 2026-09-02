from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm -- ver header completo en counter.py (paquete
# vision/palletizing/, historia acumulativa unica del modulo original).
# -----------------------------------------------------------------------
"""Mixin _LevelsMixin: asignacion de nivel (escala, apilamiento geometrico,
gravedad) y de celda, mas footprint/soporte de las cajas confirmadas."""

import logging

import numpy as np

from .formulas import (
    _build_integral_image,
    _footprint_overlap_over_min,
    _footprint_containment,
    _integral_window_sums,
    _interval_samples,
    _observed_median,
    _partial_fit_slack,
    _perspective_shrink,
    _project,
    _rasterize_rect,
    _rect_overflows_unit_square,
    _rect_support_share,
    _scale_relative_errors,
    _SupportPolygonAssessment,
    _support_polygon_assessment,
)
from .templates.template_matcher import match_layout_slot, transform_layout_template
from .templates.template_runtime import get_layout_template, get_template_capacity
from .types import LevelDecision, LevelSource

logger = logging.getLogger(__name__)


class _LevelsMixin:
    """Decision de nivel y celda; footprint y soporte de cajas confirmadas."""

    def _geometric_relative_error(
        self,
        footprint: tuple[float, float] | None,
        level: int,
    ) -> float:
        """Error absoluto de raster/localizacion relativo al lado corto.

        ``partial_fit_tolerance`` y un pixel del raster de ocupacion expresan
        errores ABSOLUTOS en el cuadrado unidad. El dominante, dividido por el
        lado corto observado, es el error relativo: la misma desviacion pesa
        mas en una caja pequena. Se limita al 20%, equivalente a exigir que el
        lado medido tenga al menos cinco unidades de error resoluble; por
        debajo no se justifica seguir relajando una decision fisica.

        Es la fuente unica de tolerancia relativa para soporte, contencion y
        matching. No interviene la clase ni ningun nombre de producto.
        """
        measured = footprint
        if measured is None or min(measured) <= 0.0:
            measured = self._level_footprint.get(level)
        if measured is None or min(measured) <= 0.0:
            return 0.0

        raster_error = 1.0 / max(int(self._cfg.occupancy_grid), 1)
        localization_error = max(
            float(getattr(self._cfg, "partial_fit_tolerance", 0.02)), 0.0,
        )
        absolute_error = max(raster_error, localization_error)
        return min(absolute_error / min(measured), 0.20)

    def _support_threshold_values(
        self,
        footprint: tuple[float, float] | None,
        level: int,
    ) -> tuple[float, float]:
        """Deriva (cobertura, ratio) desde incertidumbre geometrica relativa."""
        relative_error = self._geometric_relative_error(footprint, level)
        return (
            max(0.0, self._cfg.min_support_coverage - 0.5 * relative_error),
            self._cfg.max_support_ratio + relative_error,
        )

    def _min_redetection_containment(
        self,
        footprint: tuple[float, float] | None,
        level: int,
    ) -> float:
        """Fraccion minima de una deteccion que debe caer dentro de su identidad.

        Se permiten dos bandas de error relativo: una por cada borde opuesto
        afectado por localizacion/raster. Un fragmento ocluido puede
        encogerse, pero no extenderse fuera del rectangulo confirmado mas que
        por esa incertidumbre combinada.
        """
        return max(
            0.0,
            1.0 - 2.0 * self._geometric_relative_error(footprint, level),
        )

    # -- Seccion 5: nivel a partir de la escala aparente -----------------------
    def _assign_level(self, scale: float) -> tuple[int | None, str]:
        cfg = self._cfg
        rel_errors = _scale_relative_errors(scale, self._ladder)
        z_best = min(range(cfg.levels), key=lambda z: rel_errors[z])

        if rel_errors[z_best] <= cfg.tau_rung:
            return z_best, ""
        if scale < self._ladder[0] * (1.0 - cfg.tau_rec):
            return None, "recorte"  # oclusion, no un nivel (5.C)
        return None, "fuera-rango"  # escala fuera de la escalera S

    def _nearest_level(self, scale: float) -> int:
        """Nivel mas cercano sin rechazo; respaldo exclusivo del modo auto."""
        return min(range(self._cfg.levels), key=lambda z: abs(scale - self._ladder[z]))

    def _observe_footprint(self, key: tuple[int, int], measured: tuple[float, float]) -> None:
        """Fija una vez el footprint de la identidad espacial ``(g, z)``."""
        self._footprint.setdefault(key, measured)

    def _recompute_level_footprint(self, level: int) -> None:
        """Mediana robusta de lado largo/corto para todas las cajas del nivel.

        NO se congela cuando el nivel llega a su capacidad. Se intento y
        rompio el apilamiento: los dos llamadores invocan esto JUSTO DESPUES
        de cambiar la composicion del nivel (el bootstrap al promover celdas
        y agregar reconstruidas/ocultas; el conteo al confirmar una caja que
        puede ser la que completa el nivel), de modo que "ya esta lleno"
        siempre era cierto y el consenso se quedaba con los tamanos previos.
        Con el consenso viejo `_canonical_footprint` devolvia footprints
        desactualizados, `_has_interlocked_support` calculaba mal la
        cobertura, una caja realmente apilada no alcanzaba el umbral de
        soporte, caia al piso y `_assign_cell` la fusionaba con la celda de
        abajo que solapaba -- dos cajas fisicas compartiendo identidad y
        alternando orientacion frame a frame.
        """
        footprints = [fp for (_g, z), fp in self._footprint.items() if z == level]
        if not footprints:
            return
        long_sides = [max(fp) for fp in footprints]
        short_sides = [min(fp) for fp in footprints]
        self._level_footprint[level] = (
            _observed_median(long_sides),
            _observed_median(short_sides),
        )

    def _canonical_footprint(self, key: tuple[int, int]) -> tuple[float, float]:
        """Footprint estable usado por TODA la geometria persistente.

        Un bbox inicial inflado no puede hacer que matching vea una caja más
        grande que la que muestra el ISO. Se usa el tamaño canónico del nivel
        conservando la orientación propia de la celda.
        """
        own = self._footprint[key]
        consensus = self._level_footprint.get(key[1])
        if consensus is None:
            return own
        long_side, short_side = consensus

        # §19: cuando ambos lados difieren menos que el error geométrico, la
        # caja es cuadrada a la resolución disponible. Conservar H/V en ese
        # caso inventaría una orientación que el footprint NO puede medir y
        # haría que dos detecciones equivalentes alternen dimensiones.
        square_tolerance = self._geometric_relative_error(consensus, key[1])
        if abs(long_side - short_side) <= square_tolerance * short_side:
            return long_side, short_side
        return (long_side, short_side) if own[0] >= own[1] else (short_side, long_side)

    def _is_partial_footprint(self, footprint: tuple[float, float], level: int) -> bool:
        """True si algun lado es demasiado corto contra la mediana del nivel."""
        samples = sum(1 for (_g, z) in self._footprint if z == level)
        if samples < 3:
            return False  # el consenso aun no es robusto contra el orden de llegada
        consensus = self._level_footprint.get(level)
        if consensus is None:
            return False
        long_side, short_side = max(footprint), min(footprint)
        expected_long, expected_short = consensus
        ratio = getattr(self._cfg, "min_complete_side_ratio", 0.70)
        return (long_side < expected_long * ratio
                or short_side < expected_short * ratio)

    def _complete_partial_footprint(
        self, u: float, v: float, measured: tuple[float, float], level: int,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Reconstruye la caja COMPLETA de la que este fragmento es la parte
        visible, o None si no se puede deducir sin ambiguedad.

        Un recorte no dice donde termina la caja, pero el tamano canonico del
        nivel si dice cuanto mide. La caja completa es entonces un rectangulo
        de ese tamano que contiene al fragmento: quedan como mucho dos
        posiciones por eje (pegar el fragmento a un borde o al otro, o sea
        crecer hacia un lado o hacia el otro).

        Entre esas candidatas manda la fisica: dos cajas del mismo nivel
        estan lado a lado, nunca encimadas, asi que la correcta es la que NO
        pisa a las vecinas ya confirmadas -- crece hacia el lado libre. Si dos
        direcciones empatan, la observacion no alcanza para decidir y no se
        deduce nada: preferible una caja de menos que una inventada en el
        lugar equivocado.
        """
        consensus = self._level_footprint.get(level)
        if consensus is None:
            return None
        long_side, short_side = consensus
        du, dv = measured

        limit = getattr(self._cfg, "max_same_level_overlap", 0.10)
        tolerance = getattr(self._cfg, "partial_fit_tolerance", 0.02)
        best: tuple[tuple[float, float], tuple[float, float]] | None = None
        best_overlap = float("inf")
        equally_good: list[tuple[float, float]] = []
        # Las DOS orientaciones. Un fragmento no revela como esta girada la
        # caja entera: recortada, una caja vertical puede medir mas ancho que
        # alto. Deducir la orientacion del fragmento descartaba justamente los
        # recortes que hay que reconstruir.
        for full in ((long_side, short_side), (short_side, long_side)):
            width, height = full
            # Margen de ruido: el lado medido de un recorte puede pasarse unos
            # milimetros de la mediana sin dejar de ser un fragmento. Mas alla
            # de eso, esta orientacion simplemente no explica la observacion.
            if du > width * 1.25 or dv > height * 1.25:
                continue
            # Al recortar contra el lado canonico, un fragmento que ya lo
            # cubre entero deja de tener libertad en ese eje.
            span_u, span_v = min(du, width), min(dv, height)
            # El borde del recorte NO fija el borde de la caja: el etiquetado
            # tiene error y el robot no apoya con precision infinita. El
            # centro no tiene dos posiciones posibles sino un INTERVALO
            # continuo, y se explora entero -- quedarse en los dos extremos
            # exactos descartaba encajes validos por unos milimetros.
            slack_u = _partial_fit_slack(width, span_u, tolerance)
            slack_v = _partial_fit_slack(height, span_v, tolerance)
            for cu in _interval_samples(u, slack_u):
                for cv in _interval_samples(v, slack_v):
                    if not (0.0 <= cu <= 1.0 and 0.0 <= cv <= 1.0):
                        continue  # la caja completa se saldria de la paleta
                    overlap = self._max_same_level_overlap(cu, cv, full, level)
                    if overlap < best_overlap - 1e-9:
                        best, best_overlap = ((cu, cv), full), overlap
                        equally_good = [(cu, cv)]
                    elif abs(overlap - best_overlap) <= 1e-9:
                        equally_good.append((cu, cv))
        if best is None:
            return None
        if best_overlap > limit:
            return None
        # Varias posiciones empatan casi siempre: hay una MESETA de encajes
        # igual de buenos, no dos hipotesis. Solo es ambiguo cuando esa meseta
        # es tan ancha que la caja podria estar en un hueco distinto -- media
        # caja de separacion. Por debajo de eso, la respuesta es el centro de
        # la meseta, que es la mejor estimacion del centro real.
        (_cu, _cv), full = best
        spread_u = max(p[0] for p in equally_good) - min(p[0] for p in equally_good)
        spread_v = max(p[1] for p in equally_good) - min(p[1] for p in equally_good)
        if spread_u > full[0] / 2.0 or spread_v > full[1] / 2.0:
            return None
        center_u = sum(p[0] for p in equally_good) / len(equally_good)
        center_v = sum(p[1] for p in equally_good) / len(equally_good)
        return (center_u, center_v), full

    def _max_same_level_overlap(
        self, u: float, v: float, footprint: tuple[float, float], level: int,
    ) -> float:
        """Mayor interpenetracion con otra caja confirmada del mismo nivel."""
        overlaps = [
            _footprint_overlap_over_min(
                (u, v), footprint,
                self._cell_position(g, z), self._canonical_footprint((g, z)),
            )
            for (g, z) in self._occupied
            if z == level and (g, z) in self._footprint
        ]
        return max(overlaps, default=0.0)

    def _level_with_support(self, u: float, v: float, footprint: tuple[float, float], level: int) -> int:
        """Baja `level` hasta el primero con soporte estable debajo.

        Invariante fisico: una caja no flota. Para estar en el nivel z>0
        su centroide debe caer dentro del hull de los contactos del nivel z-1,
        con al menos dos apoyos independientes. Un hull degenerado usa K/phi.

        También protege las decisiones de escala que no traen evidencia de
        oclusión anclada a una identidad inferior confirmada.
        """
        tracking_floor = self._tracking_floor_level if self._cfg.layout_mode == "auto" else 0
        level = max(level, tracking_floor)
        while level > tracking_floor:
            supported = (
                self._level_is_full(level - 1)
                and self._has_interlocked_support(u, v, footprint, level - 1)
            )
            if supported:
                return level
            logger.debug(
                "nivel %d sin poligono de soporte estable en (%.3f, %.3f) "
                "sobre el nivel %d -- baja a %d",
                level, u, v, level - 1, level - 1,
            )
            level -= 1
        return level

    def _advance_tracking_floor(self) -> None:
        """Desliza la ventana local sin renumerar el estado absoluto.

        Si el nivel absoluto i queda completo, para el seguimiento se vuelve
        el piso local 0: i-1 y anteriores dejan de participar en matching,
        soporte y fallback. El ISO/JSON conservan i como nivel absoluto.
        """
        if self._cfg.layout_mode != "auto":
            return
        capacity = get_template_capacity(self._box_class or "")
        if capacity is None:
            return
        previous = self._tracking_floor_level
        completed = [
            level
            for level in range(previous + 1, self._cfg.levels)
            if sum(1 for _cell, z in self._occupied if z == level) == capacity
        ]
        if not completed:
            return
        self._tracking_floor_level = max(completed)
        logger.debug(
            "VENTANA TRACKING DESLIZADA: nivel absoluto N%d pasa a local 0; "
            "N%d pasa a local 1 y niveles absolutos < N%d se ignoran solo en seguimiento",
            self._tracking_floor_level, self._tracking_floor_level + 1,
            self._tracking_floor_level,
        )

    def _level_is_full(self, level: int) -> bool:
        """True si en el nivel `level` ya no entra otra caja.

        Regla fisica de paletizado: no se empieza el nivel i+1 mientras el
        nivel i tenga un hueco libre donde apoyar. Es una restriccion mucho
        mas fuerte que cualquier pista visual, porque no depende del
        detector ni de la calibracion: depende de donde estan las cajas.

        Se resuelve en el cuadrado unidad de la homografia, donde la paleta
        ya esta rectificada y los footprints son rectangulos alineados. Se
        rasteriza la ocupacion y se busca, con imagen integral, cualquier
        ventana libre del tamano de una caja. Se prueban las dos
        orientaciones porque las cajas se apoyan giradas 90 grados
        indistintamente.

        La ocupacion incluye las celdas de ESTE nivel y las del nivel de
        arriba: una caja apoyada encima tapa exactamente un footprint de
        este nivel, asi que el area que cubre no es hueco libre sino una
        caja que la camara nunca pudo observar. Es lo que permite dar por
        lleno un nivel cuyas cajas tapadas nunca llegaron a `_occupied`.

        Devuelve False si el nivel todavia no tiene consenso de footprint o
        no tiene ninguna celda: sin referencia de tamano no se puede afirmar
        que este lleno, y ante la duda NO se promueve.
        """
        # Capacidad declarada: la paleta esta empernada al piso, o sea es
        # fija, y las cajas son todas iguales -- entonces cuantas entran por
        # nivel es un NUMERO conocido, no algo que haya que deducir. Cuando
        # esta configurado manda sobre la geometria, que depende de que la
        # ROI este ajustada a la carga real: una ROI mas grande que la
        # paleta (lo normal al dibujarla a mano) deja hueco libre para
        # siempre y el nivel nunca se daria por lleno.
        #
        # Excepcion: ANTES de reconciliar el inventario inicial de
        # una paleta que ya vino apilada) la cuenta exacta puede ser
        # imposible de alcanzar por diseno -- una camara cenital nunca ve
        # las cajas del piso que ya nacieron tapadas por un nivel superior
        # desde el primer frame. Exigir capacity aqui las dejaba fuera de
        # `_occupied` para siempre y el nivel de arriba nunca se promovia
        # (deadlock). En bootstrap se cae al chequeo geometrico de huecos
        # libres, que no depende de haber confirmado cada celda una por
        # una. Al comenzar la operacion (`arm_cycle_seen`) tambien manda la
        # cantidad exacta: el lazo de bootstrap deja de reconciliar en ese
        # punto y no puede dejar el conteo operativo dependiendo del raster.
        capacity = get_template_capacity(self._box_class or "")
        # En operacion normal, el primer ciclo del brazo tambien cierra el
        # inventario inicial aunque `_bootstrap_reconciled` no haya podido
        # terminar. `_reconcile_initial_layers()` deja de correr al ver ese
        # ciclo, asi que usar solo `_bootstrap_reconciled` dejaba el sistema
        # atrapado en el raster geometrico: N0 podia tener exactamente la
        # capacidad (15), pero la ROI todavia dejaba un hueco aparente y
        # nunca se habilitaba N1.
        #
        # El orden es intencional: una vez que empieza la operacion manda la
        # capacidad exacta, incluso si durante bootstrap se marco un nivel
        # como `proven_full` por oclusion.
        if capacity is not None and (
            self._bootstrap_reconciled
            or (self._arm_cycle_seen and not self._template_baseline_pending)
        ):
            occupied_here = sum(1 for _g, z in self._occupied if z == level)
            return occupied_here == capacity

        # Probado lleno por oclusion: hay una caja trabada encima, y eso no
        # ocurre sobre un nivel con hueco libre. Vale aunque sus cajas tapadas
        # nunca hayan llegado a `_occupied`.
        if level in self._proven_full:
            return True

        reference = self._level_footprint.get(level)
        if reference is None:
            return False

        n = self._cfg.occupancy_grid
        occupancy = np.zeros((n, n), dtype=np.int32)
        cells = 0
        for (g, z) in self._footprint:
            if (g, z) not in self._occupied:
                continue
            du, dv = self._canonical_footprint((g, z))
            if z == level:
                pass
            elif z > level:
                # Invariante de area (docs/palletizing_math.md seccion 7.2):
                # una caja del nivel de arriba tapa EXACTAMENTE
                # un footprint de este nivel. El hueco que deja en el raster
                # no es espacio libre: es una caja que la camara nunca pudo
                # ver. Sin contarlo, una paleta que arranca ya apilada jamas
                # daba el nivel por lleno -- las cajas tapadas dejaban huecos
                # permanentes -- y el nivel superior no se podia promover.
                #
                # Su footprint viene inflado por perspectiva: esa caja esta
                # mas cerca de la camara y la homografia rectifica el plano
                # del nivel 0, no el suyo. Se lo encoge a la escala de ESTE
                # nivel con la razon de peldanos de la escalera, si no tapa
                # mas area de la que fisicamente le toca y el nivel se daria
                # por lleno antes de tiempo.
                du, dv = _perspective_shrink(du, dv, self._ladder, z, level)
            else:
                continue
            u, v = self._cell_position(g, z)
            u0, u1, v0, v1 = _rasterize_rect(u, v, du, dv, n)
            if u1 > u0 and v1 > v0:
                occupancy[v0:v1, u0:u1] = 1
                cells += 1
        if cells == 0:
            return False

        integral = _build_integral_image(occupancy)
        for side_u, side_v in ((reference[0], reference[1]), (reference[1], reference[0])):
            w = int(side_u * n * self._cfg.free_gap_ratio)
            h = int(side_v * n * self._cfg.free_gap_ratio)
            if w < 1 or h < 1 or w > n or h > n:
                continue
            # Suma de cada ventana h x w: 0 => completamente libre.
            window_sums = _integral_window_sums(integral, w, h)
            if bool((window_sums == 0).any()):
                return False
        return True

    def _is_stack_candidate(
        self,
        idx: int,
        matched: dict[int, tuple[int, int]],
        u: float,
        v: float,
        footprint: tuple[float, float],
    ) -> bool:
        """True si la deteccion `idx`, emparejada por solape con una celda ya
        confirmada, es en realidad una caja APOYADA ENCIMA de esa celda y no
        una nueva vista de ella.

        Solape alto por si solo no distingue los dos casos: una caja encima de
        la celda 7 pisa la celda 7 tanto como la propia celda 7. Hacen falta
        las tres condiciones juntas.
        """
        cell, level = matched[idx]
        if (
            not self._arm_cycle_seen
            or self._template_baseline_pending
            or self._placement_credits <= 0
        ):
            return False  # sin viaje terminado no existe una caja nueva fisicamente
        if level + 1 >= self._cfg.levels:
            return False
        if self._cell_frame.get((cell, level)) == self._current_frame:
            return False  # confirmada en este mismo frame: nada pudo apilarse aun
        below = self._footprint.get((cell, level))
        if below is None:
            return False
        area = footprint[0] * footprint[1]
        # Mas chica que la de abajo -> recorte por oclusion, no una caja encima.
        if area < below[0] * below[1] * self._cfg.min_stack_area_ratio:
            return False
        # Y sobre todo: no se apila sobre un nivel que todavia tiene hueco.
        if not self._level_is_full(level):
            logger.debug(
                "no se apila sobre celda=%d nivel=%d: el nivel todavia tiene hueco libre",
                cell, level,
            )
            return False
        return self._has_interlocked_support(u, v, footprint, level)

    def _log_new_candidate(
        self, u: float, v: float, footprint: tuple[float, float]
    ) -> None:
        """Vuelca el analisis completo de una deteccion que NO corresponde a
        ninguna celda conocida -- una caja que antes no estaba.

        Son las unicas que pueden cambiar el conteo, asi que son las unicas
        que vale la pena mirar en detalle. Para cada nivel se lista cuanto lo
        sostiene y que celdas aportan; con eso se ve por que termino en el
        nivel que termino, sin tener que deducirlo del resultado.

        Solo se invoca para candidatos que pueden cambiar ocupacion.
        """
        diagnostic_key = (
            round(u, 2), round(v, 2),
            round(footprint[0], 2), round(footprint[1], 2),
        )
        if diagnostic_key in self._logged_candidate_diagnostics:
            return
        self._logged_candidate_diagnostics.add(diagnostic_key)

        logger.debug(
            "caja nueva a analizar: pos=(%.3f, %.3f) tam=%.3fx%.3f",
            u, v, footprint[0], footprint[1],
        )
        for z in range(self._cfg.levels):
            min_coverage, max_ratio = self._support_threshold_values(footprint, z + 1)
            supporters, coverage = self._best_support(u, v, footprint, z)
            if not supporters:
                continue
            detalle = ", ".join(f"celda={g}:{share * 100:.0f}%" for g, share in supporters)
            assessment = self._support_polygon(u, v, footprint, z)
            # Mismas ramas que _has_interlocked_support. Un diagnóstico que
            # todavía explique top-2 mientras el runtime usa hull sería PEOR
            # que no registrar nada: enseñaría una causa falsa al operador.
            faltas = []
            if assessment.contact_count < 2:
                faltas.append(f"apoyos independientes ({assessment.contact_count} de >=2)")
            elif sum(assessment.shares) > 1.0 + 0.5 * self._geometric_relative_error(footprint, z + 1):
                faltas.append("soportes inferiores interpenetrados/duplicados")
            elif assessment.degenerate:
                fallback_ok = (
                    coverage >= min_coverage
                    and self._dynamic_support_is_balanced(
                        list(assessment.shares), min_coverage, max_ratio,
                    )
                )
                if not fallback_ok:
                    faltas.append(
                        "hull degenerado y fallback K/phi insuficiente "
                        f"(cobertura={coverage * 100:.0f}%)"
                    )
            elif not assessment.center_inside:
                faltas.append(
                    "centroide fuera del poligono de soporte "
                    f"(distancia={assessment.center_distance:.4f})"
                )
            veredicto = "APILADA" if not faltas else "no apila -- falta: " + ", ".join(faltas)
            logger.debug(
                "  nivel %d: hull=%.0f%% cobertura=%.0f%% apoyos=%d [%s] lleno=%s -> %s",
                z, assessment.hull_area_ratio * 100.0, coverage * 100.0,
                assessment.contact_count, detalle,
                self._level_is_full(z), veredicto,
            )

    def _supporters(
        self, u: float, v: float, footprint: tuple[float, float], level: int
    ) -> list[tuple[int, float]]:
        """Celdas del nivel `level` que intersectan este footprint,
        como (celda, fraccion), de mayor a menor.

        No existe umbral minimo individual: `tau_cell_overlap` pertenece al
        matching de identidad y NO a la fisica de soporte. Se conservan todos
        los aportes positivos y la decision posterior usa solo los dos
        mayores. Se excluyen las celdas confirmadas en este mismo frame."""
        shares = []
        for (g, z) in self._footprint:
            if z != level or (g, z) not in self._occupied:
                continue
            if self._cell_frame.get((g, z)) == self._current_frame:
                continue
            fp = self._canonical_footprint((g, z))
            cu, cv = self._cell_position(g, z)
            du, dv = footprint
            fu, fv = fp
            # Fraccion de LA CAJA SUPERIOR sostenida por esta inferior. Antes
            # se dividia por el area menor y una esquina diminuta podia valer
            # 100% si el soporte tambien era pequeno.
            share = _rect_support_share(u, v, du, dv, cu, cv, fu, fv)
            if share > 0.0:
                shares.append((g, share))
        shares.sort(key=lambda item: item[1], reverse=True)
        return shares

    def _best_support(
        self, u: float, v: float, footprint: tuple[float, float], level: int,
    ) -> tuple[list[tuple[int, float]], float]:
        """Prefijo mínimo K cuya unión alcanza la cobertura de soporte.

        Antes se cortaba en cuatro y el balance lo decidía top-2. Eso era
        correcto para una rejilla de cajas iguales, pero NO para tamaños
        mixtos. Ahora esta función solo descubre K y cobertura; el camino
        principal usa el hull y el fallback evalúa ``phi`` sobre ese K.
        """
        supporters = self._supporters(u, v, footprint, level)
        min_coverage, _max_ratio = self._support_threshold_values(footprint, level + 1)
        if len(supporters) < 2:
            return supporters, 0.0
        top, coverage = supporters[:2], 0.0
        # §11: K es el número mínimo de contactos cuya UNIÓN alcanza la
        # cobertura requerida. No existe K_max=4: tamaños mixtos pueden
        # cruzar más celdas y cortar en cuatro reintroduciría el supuesto que
        # precisamente se está eliminando.
        for count in range(2, len(supporters) + 1):
            top = supporters[:count]
            coverage = self._support_coverage(
                u, v, footprint, level,
                support_cells={cell for cell, _share in top},
            )
            if coverage >= min_coverage:
                break
        return top, coverage

    @staticmethod
    def _dynamic_support_is_balanced(
        shares: list[float], min_coverage: float, max_ratio: float,
    ) -> bool:
        """Fallback K/phi de §11, sin techo K ni parámetros por catálogo.

        ``K`` es el prefijo mínimo que alcanza la cobertura. El caso K=1 se
        rechaza explícitamente: estabilidad sobre una caja no es amarre. Para
        K=2, ``phi <= rho/(rho+1)`` es algebraicamente equivalente al ratio
        histórico ``s1/s2 <= rho``; para K>2 generaliza el mismo concepto de
        que un único soporte no domine la masa sostenida.
        """
        positive = sorted((float(s) for s in shares if s > 0.0), reverse=True)
        cumulative = 0.0
        selected: list[float] = []
        for share in positive:
            selected.append(share)
            cumulative += share
            if cumulative >= min_coverage:
                break
        if len(selected) < 2 or cumulative < min_coverage:
            return False
        phi = selected[0] / max(cumulative, 1e-9)
        return phi <= max_ratio / (max_ratio + 1.0)

    def _support_polygon(
        self, u: float, v: float, footprint: tuple[float, float], level: int,
    ) -> _SupportPolygonAssessment:
        """Construye el criterio primario §12 con todos los apoyos válidos."""
        supports = []
        for (g, z) in self._footprint:
            if z != level or (g, z) not in self._occupied:
                continue
            if self._cell_frame.get((g, z)) == self._current_frame:
                continue
            cu, cv = self._cell_position(g, z)
            fu, fv = self._canonical_footprint((g, z))
            supports.append((cu, cv, fu, fv))

        relative_error = self._geometric_relative_error(footprint, level + 1)
        # Se reutiliza una sola fuente de incertidumbre, sin nuevos valores en
        # YAML: media banda para slivers/hull y margen cero para no rechazar
        # un centroide exactamente sobre una frontera ideal compartida.
        noise_floor = 0.5 * relative_error
        return _support_polygon_assessment(
            (u, v, footprint[0], footprint[1]), supports,
            min_contact_ratio=noise_floor,
            min_hull_area_ratio=noise_floor,
            center_margin_ratio=0.0,
        )

    def _has_interlocked_support(
        self, u: float, v: float, footprint: tuple[float, float], level: int,
    ) -> bool:
        """Polígono de soporte primario; K/phi solo para hull degenerado.

        No basta que el centro esté sostenido: el conteo exige dos contactos
        independientes para no convertir una redetección 1-a-1 en caja nueva.
        """
        assessment = self._support_polygon(u, v, footprint, level)
        if assessment.interlocked:
            return True
        if assessment.contact_count < 2 or not assessment.degenerate:
            return False

        # §13: solo un hull casi plano autoriza el fallback de áreas. En un
        # hull sano con centro fuera, caer a K/phi escondería una inestabilidad
        # física real y produciría un falso positivo.
        min_coverage, max_ratio = self._support_threshold_values(footprint, level + 1)
        coverage = self._support_coverage(u, v, footprint, level)
        return (
            coverage >= min_coverage
            and self._dynamic_support_is_balanced(
                list(assessment.shares), min_coverage, max_ratio,
            )
        )

    def _supporters_are_balanced(
        self,
        shares: list[float],
        footprint: tuple[float, float],
        level: int,
    ) -> bool:
        """Evita confundir coincidencia 1-a-1 con amarre entre dos cajas.

        Para s1 >= s2 > 0 se exige s1/s2 <= max_support_ratio: un 90/10 es
        coincidencia con una caja, no un amarre; 32/30 si representa dos
        apoyos comparables. Se llama SIEMPRE con los 2 apoyos principales
        (nunca con un 3ro/4to residual, ver `_best_support`): un derrame
        pequeño hacia una celda vecina no vuelve falso un amarre ya
        confirmado entre los dos principales. La cobertura conjunta se
        valida por separado: por eso 1/1 no pasa aunque sea parejo.
        """
        if len(shares) < 2 or shares[1] <= 0.0:
            return False
        _min_coverage, max_ratio = self._support_threshold_values(footprint, level)
        return self._support_balance_ratio(shares, footprint, level) <= max_ratio

    def _support_balance_ratio(
        self,
        shares: list[float],
        footprint: tuple[float, float],
        level: int,
    ) -> float:
        """Ratio top-2 descontando incertidumbre del borde compartido.

        La localizacion puede trasladar area aparente del segundo apoyo al
        primero. Se corrige simetricamente una cuarta parte del error relativo
        (un borde de los cuatro del rectangulo), sin convertir un contacto
        residual 90/10 en un amarre valido.
        """
        if len(shares) < 2 or shares[1] <= 0.0:
            return float("inf")
        uncertainty = 0.50 * self._geometric_relative_error(footprint, level)
        major = max(0.0, shares[0] - uncertainty)
        minor = shares[1] + uncertainty
        return major / max(minor, 1e-9)

    def _support_coverage(
        self,
        u: float,
        v: float,
        footprint: tuple[float, float],
        level: int,
        support_cells: set[int] | None = None,
    ) -> float:
        """Fraccion del footprint (u, v, `footprint`) que esta sostenida por
        la UNION de las celdas confirmadas del nivel `level`.

        Se rasteriza igual que `_level_is_full`, en el cuadrado unidad de la
        homografia donde la paleta ya esta rectificada. Devuelve 0.0 si el
        nivel no tiene celdas utiles.

        Se excluyen las celdas confirmadas en ESTE mismo frame: nada pudo
        apoyarse sobre ellas todavia, y si dos detecciones del mismo frame se
        pisan son la misma caja vista dos veces, no una sobre otra.
        """
        n = self._cfg.occupancy_grid
        support = np.zeros((n, n), dtype=bool)
        cells = 0
        for (g, z) in self._footprint:
            if z != level or (g, z) not in self._occupied:
                continue
            if support_cells is not None and g not in support_cells:
                continue
            if self._cell_frame.get((g, z)) == self._current_frame:
                continue
            du, dv = self._canonical_footprint((g, z))
            cu, cv = self._cell_position(g, z)
            u0, u1, v0, v1 = _rasterize_rect(cu, cv, du, dv, n)
            if u1 > u0 and v1 > v0:
                support[v0:v1, u0:u1] = True
                cells += 1
        if cells == 0:
            return 0.0

        du, dv = footprint
        bu0, bu1, bv0, bv1 = _rasterize_rect(u, v, du, dv, n)
        if bu1 <= bu0 or bv1 <= bv0:
            return 0.0
        window = support[bv0:bv1, bu0:bu1]
        return float(window.sum()) / float(window.size)

    def _stacking_level(
        self, u: float, v: float, footprint: tuple[float, float]
    ) -> int | None:
        """Nivel deducido por GEOMETRIA, no por escala aparente.

        Una detección está apilada sobre el nivel z cuando el centroide cae
        dentro del polígono convexo formado por TODOS sus contactos válidos y
        existen al menos dos soportes independientes. Solo un hull degenerado
        cae al acumulador K/phi de cobertura y balance.

        (1) sola no distingue nada: una re-deteccion tambien esta
        "sostenida al 100%" por la celda que ella misma ocupa.

        Se prefiere todo esto a la escalera s(z) porque no depende de c_z ni
        de reference_scale_px -- solo de donde estan las cajas y de que se
        ve de ellas. Devuelve None si nada la sostiene (piso, nivel 0).
        """
        # Se busca el nivel MAS ALTO que sostenga esta deteccion. El
        # criterio es el hull de TODOS los contactos válidos, no cuánto pisa
        # a una vecina particular. Esto permite dos, tres o más soportes sin
        # fijar K_max. K/phi solo decide cuando el hull casi plano no permite
        # un test punto-en-polígono numéricamente confiable.
        supported_level = None
        for z in range(self._cfg.levels - 1, self._tracking_floor_level - 1, -1):
            if self._has_interlocked_support(u, v, footprint, z):
                supported_level = z
                break
        if supported_level is None:
            return None
        supporters, coverage = self._best_support(u, v, footprint, supported_level)

        z_base = supported_level
        g_base = supporters[0][0] if supporters else -1
        # Regla de paletizado: el nivel i+1 no empieza mientras el nivel i
        # tenga donde apoyar. Si todavia hay hueco, esto no puede estar
        # encima -- es la misma caja vista de nuevo, o un bbox duplicado.
        if not self._level_is_full(z_base):
            if (g_base, z_base) not in self._rejected_in_cycle:
                self._rejected_in_cycle.add((g_base, z_base))
                logger.debug(
                    "no se apila: la deteccion esta sostenida al %.0f%% por el nivel %d "
                    "(celda=%d), pero ese nivel aun tiene hueco libre -- se trata como la "
                    "misma caja, no una encima",
                    coverage * 100.0, z_base, g_base,
                )
            return z_base

        if z_base + 1 >= self._cfg.levels:
            self._log.warning(
                "apilamiento: deteccion sostenida por el nivel %d (celda=%d, cobertura %.0f%%) "
                "pediria nivel %d, "
                "pero solo hay %d niveles configurados -- se queda en %d",
                z_base, g_base, coverage * 100.0, z_base + 1, self._cfg.levels, z_base,
            )
            return z_base

        logger.debug(
            "apilamiento: deteccion sostenida al %.0f%% por el nivel %d, "
            "apoyada en %d celda(s) [%s] "
            "-> se le asigna nivel %d",
            coverage * 100.0, z_base, len(supporters),
            ", ".join(f"celda={g}:{share * 100:.0f}%" for g, share in supporters) or "ninguna",
            z_base + 1,
        )
        return z_base + 1

    # -- Puerta unica de decision de nivel -------------------------------------
    def _resolve_level(
        self,
        cx: float,
        cy: float,
        footprint: tuple[float, float],
        ladder_level: int,
        from_occlusion: bool,
    ) -> LevelDecision:
        """Resuelve el nivel de UNA deteccion y deja dicho quien lo decidio.

        Orden de autoridad, de mayor a menor:

          1. `from_occlusion` -- el override por oclusion ya comparo dos
             detecciones del mismo frame. Es la unica evidencia que ve el
             apilado inicial de un video que arranca con la paleta cargada,
             asi que si disparo, manda y no se toca.
          2. geometria de apilamiento (`_stacking_level`) -- que una caja
             pise a otra es observacion directa.
          3. la escalera s(z) -- depende de c_z y reference_scale_px, que se
             calibran a mano y no se pueden verificar en vivo. Solo queda si
             el modo no es `auto`.

        Y despues, siempre, el filtro de gravedad: una caja no flota.
        """
        if from_occlusion:
            # El soporte de la de arriba es la OTRA deteccion del mismo frame,
            # que todavia no esta confirmada -- por eso tampoco se le aplica
            # el filtro de gravedad, que solo mira celdas ya confirmadas.
            return LevelDecision(ladder_level, LevelSource.OCCLUSION)

        u, v = _project(self._homography, cx, cy)
        if self._cfg.layout_mode != "auto":
            grounded = self._level_with_support(u, v, footprint, ladder_level)
            source = LevelSource.LADDER if grounded == ladder_level else LevelSource.GRAVITY
            return LevelDecision(grounded, source)

        # Llego aca porque no se emparejo con ninguna celda conocida: es una
        # caja que antes no estaba. Se vuelca el analisis completo antes de
        # decidir, no despues.
        self._log_new_candidate(u, v, footprint)

        stacked = self._stacking_level(u, v, footprint)
        # None NO significa "sin opinion": significa que no pisa ninguna
        # celda, o sea que esta apoyada en el piso -> nivel 0. Dejar el nivel
        # de la escalera en ese caso era el bug que ponia cajas sueltas en el
        # nivel 1 (la escalera esta mal calibrada y no separa niveles; ver
        # _check_ladder_is_separable).
        if stacked is None:
            level, source = self._tracking_floor_level, LevelSource.FLOOR
        else:
            level, source = stacked, LevelSource.STACKING
        if level != ladder_level:
            logger.debug(
                "nivel corregido por %s: escalera decia %s, geometria dice %d",
                source.value, ladder_level, level,
            )

        # Una caja no flota. Si el nivel asignado no tiene una celda
        # confirmada debajo que lo sostenga, se baja hasta el que si la tenga.
        if level > 0:
            grounded = self._level_with_support(u, v, footprint, level)
            if grounded != level:
                level, source = grounded, LevelSource.GRAVITY
        return LevelDecision(level, source)

    # -- Seccion 4/8.B: celda mas cercana a la posicion proyectada -------------
    def _assign_cell(
        self,
        cx: float,
        cy: float,
        level: int,
        footprint: tuple[float, float] | None = None,
        pixel_footprint: tuple[float, float] | None = None,
        *,
        reuse_occupied: bool = True,
    ) -> tuple[int | None, str]:
        u, v = _project(self._homography, cx, cy)
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return None, "fuera-paleta"

        if self._cfg.layout_mode == "auto":
            # Solo celdas CONFIRMADAS. Una posicion descubierta cuya
            # deteccion despues fue rechazada (por ejemplo, por el limite de
            # una caja por ciclo de brazo) queda viva en _dynamic_positions a
            # proposito, para conservar un identificador espacial estable.
            # Pero no es una caja, asi que no puede reclamar
            # detecciones futuras: una caja apoyada sobre OTRA celda caia
            # dentro de tau_cell de esa posicion fantasma, heredaba su id y
            # el ISO la dibujaba donde no estaba.
            known = [
                (g, pos) for (g, z), pos in self._dynamic_positions.items()
                if z == level and (g, z) in self._occupied
            ] if reuse_occupied else []

            # Criterio PRIMARIO: area de interseccion del footprint, no
            # distancia de centroide. La distancia sola se confunde cuando
            # el tamano etiquetado/medido varia entre ejemplos -- dos cajas
            # bien separadas pueden tener centroides mas cerca entre si que
            # una caja consigo misma vista con el bbox recortado por
            # oclusion (eso corre el centroide hasta ~1/4 del lado largo).
            # El area de interseccion no tiene ese problema: pregunta
            # directamente "esto pisa el mismo lugar" en vez de "esto queda
            # cerca", que es lo que en verdad define si es la misma caja.
            overlap, containment, g_overlapped = 0.0, 0.0, -1
            if footprint is not None and known:
                candidates = []
                for g, pos in known:
                    confirmed = self._canonical_footprint((g, level))
                    candidates.append((
                        _footprint_overlap_over_min((u, v), footprint, pos, confirmed),
                        _footprint_containment((u, v), footprint, pos, confirmed),
                        g,
                    ))
                overlap, containment, g_overlapped = max(candidates)
                min_containment = self._min_redetection_containment(footprint, level)
                if (
                    overlap >= self._cfg.tau_cell_overlap
                    and containment >= min_containment
                ):
                    logger.debug(
                        "celda reusada: deteccion en (%.3f, %.3f) solapa %.0f%% y "
                        "queda contenida %.0f%% en celda=%d nivel=%d (min %.0f%%)",
                        u, v, overlap * 100.0, containment * 100.0,
                        g_overlapped, level, min_containment * 100.0,
                    )
                    return g_overlapped, ""

            # Respaldo por distancia: solo entra si no hay footprint medido
            # (no deberia pasar en uso normal, `update()` siempre lo mide).
            # Si existe footprint, un overlap insuficiente es evidencia de
            # una caja fisica distinta aunque su centro quede dentro de
            # tau_cell; volver a distancia fusionaria cajas vecinas.
            if footprint is None and known:
                g_best, (best_u, best_v) = min(
                    known,
                    key=lambda item: float(np.hypot(u - item[1][0], v - item[1][1])),
                )
                if float(np.hypot(u - best_u, v - best_v)) <= self._cfg.tau_cell:
                    return g_best, ""

            # Para productos con patron fisico calibrado, ``cell`` deja de
            # depender del orden en que YOLO encontro las cajas. Se asigna
            # al rectangulo A/B que ocupa realmente. Asi los huecos
            # conservan identidad aunque el proceso arranque con la paleta
            # parcialmente cargada o con cajas inferiores ocluidas.
            phase_candidates = (
                (self._template_phase,)
                if self._template_phase is not None
                else (0, 1)
            )
            templates = [
                transform_layout_template(template, self._template_registration)
                for phase in phase_candidates
                if (template := get_layout_template(
                    self._box_class or "", level, phase,
                )) is not None
            ]
            if (
                templates
                and footprint is not None
                and pixel_footprint is not None
                and self._template_registration is not None
            ):
                template_tolerance = self._cfg.tau_cell * max(
                    self._template_registration[0],
                    self._template_registration[1],
                )
                occupied_cells = {
                    g for g, z in self._occupied if z == level
                } if not reuse_occupied else set()
                matches = [
                    (float(np.hypot(cx - slot.u, cy - slot.v)), template, slot)
                    for template in templates
                    if (slot := match_layout_slot(
                        template,
                        center=(cx, cy),
                        footprint=pixel_footprint,
                        occupied=occupied_cells,
                        max_center_distance=template_tolerance,
                    )) is not None
                ]
                matched = min(matches, default=None, key=lambda item: item[0])
                slot = None if matched is None else matched[2]
                if slot is None:
                    logger.debug(
                        "deteccion fuera de plantilla: clase=%s nivel=%d pos=(%.3f, %.3f) "
                        "tam=%.3fx%.3f",
                        self._box_class, level, cx, cy,
                        pixel_footprint[0], pixel_footprint[1],
                    )
                    return None, "fuera-plantilla"
                logger.debug(
                    "hueco predicho: clase=%s patron=%s nivel=%d celda=%d "
                    "observado=(%.3f, %.3f) esperado=(%.3f, %.3f) orientacion=%s",
                    self._box_class,
                    "A" if matched[1].pattern == 0 else "B",
                    level, slot.cell, cx, cy, slot.u, slot.v, slot.orientation.value,
                )
                return slot.cell, ""

            g_new = self._next_cell_by_level.get(level, 0)
            logger.debug(
                "candidata de celda nueva: id tentativo=%d nivel=%d en (%.3f, %.3f) -- "
                "max overlap=%.0f%% contencion=%.0f%% (celda=%d)",
                g_new, level, u, v, overlap * 100.0, containment * 100.0,
                g_overlapped,
            )
            return g_new, ""

        cells = self._cfg.levels_layout[level].cells  # posiciones declaradas de este nivel, no una rejilla

        distances = [float(np.hypot(u - cu, v - cv)) for cu, cv in cells]
        g_best = min(range(len(cells)), key=lambda g: distances[g])

        if distances[g_best] > self._cfg.tau_cell:
            return None, "F2"  # residuo a la celda mas cercana > tau_cell

        return g_best, ""
