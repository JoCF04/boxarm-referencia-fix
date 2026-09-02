from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : Los tests siguen al cerebro: update() recibe FrameInput (con
#              el gate adentro) y la escena se inspecciona por scene_state()
#              en vez de occupied_cells()/cell_position(). Se absorbe aqui
#              test_isometric_uniform_footprint.py, porque la unificacion de
#              footprint por nivel dejo de vivir en el renderizador.
# -----------------------------------------------------------------------

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from boxarm.config import load_palletizing_config
from boxarm.vision.palletizing import (
    CellState,
    FrameInput,
    GateState,
    GridDetection,
    GridCounter,
    _observed_median,
)
from boxarm.vision.palletizing.formulas import _support_polygon_assessment
from boxarm.vision.drawing import draw_grid_detections


def test_observed_median_never_averages_two_box_sizes() -> None:
    assert _observed_median([0.20, 0.40]) == 0.40


def test_productive_support_calibration_accepts_measured_homography_margin() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    cfg = load_palletizing_config(config_path)

    # No existen politicas por SKU: los defaults son la asintota estricta y
    # la tolerancia efectiva sale unicamente del tamano observado.
    assert cfg.min_support_coverage == 0.75
    assert cfg.max_support_ratio == 2.0
    assert not hasattr(cfg, "class_thresholds")

    counter = GridCounter(
        np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32),
        cfg,
    )
    small_coverage, small_ratio = counter._support_threshold_values(
        (0.173, 0.10), level=0,
    )
    assert np.isclose(small_coverage, 0.65)
    assert np.isclose(small_ratio, 2.2)
    assert small_ratio >= 55 / 26

    large_coverage, large_ratio = counter._support_threshold_values(
        (0.40, 0.40), level=0,
    )
    assert large_coverage > small_coverage
    assert large_ratio < small_ratio
    assert np.isclose(large_coverage, 0.725)
    assert np.isclose(large_ratio, 2.05)

    counter._level_footprint[0] = (0.40, 0.40)
    assert counter._support_threshold_values(None, level=0) == (
        large_coverage, large_ratio,
    )


def test_support_methods_adapt_to_geometry_without_consulting_box_class() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    cfg = load_palletizing_config(config_path)
    roi = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
    counter = GridCounter(roi, cfg)

    counter.set_box_class("coin_roll_100")
    small_thresholds = counter._support_threshold_values((0.173, 0.10), level=0)
    large_thresholds = counter._support_threshold_values((0.40, 0.40), level=0)
    assert counter._dynamic_support_is_balanced(
        [0.55, 0.26], *small_thresholds,
    )
    assert not counter._dynamic_support_is_balanced(
        [0.55, 0.26], *large_thresholds,
    )

    # Cambiar solo el nombre no puede cambiar los umbrales efectivos.
    before = counter._support_threshold_values((0.173, 0.10), level=0)
    counter.set_box_class("coin_roll_10")
    after = counter._support_threshold_values((0.173, 0.10), level=0)
    assert before == after


def test_support_balance_accepts_real_offset_but_rejects_tiny_secondary_contact() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    cfg = load_palletizing_config(config_path)
    counter = GridCounter(
        np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32),
        cfg,
    )
    small = (0.178, 0.100)

    # Casos productivos del log: la segunda caja sostiene 23%; no es un
    # contacto residual aunque el cociente crudo 55/23 supere 2.20.
    assert counter._supporters_are_balanced([0.55, 0.23], small, level=1)
    # Ultima caja real del log: cobertura 79%, apoyos 58%/17%.
    assert counter._supporters_are_balanced([0.58, 0.17], (0.107, 0.162), level=1)
    # Una casi coincidencia uno-a-uno sigue sin ser paletizado trabado.
    assert not counter._supporters_are_balanced([0.90, 0.10], small, level=1)


def test_support_polygon_rejects_one_support_even_when_it_contains_centroid() -> None:
    """Estabilidad local sobre una caja no equivale a paletizado trabado."""
    assessment = _support_polygon_assessment(
        target=(0.50, 0.50, 0.40, 0.40),
        supports=[(0.45, 0.50, 0.40, 0.40)],
        min_contact_ratio=0.01,
        min_hull_area_ratio=0.01,
        center_margin_ratio=0.0,
    )

    assert assessment.contact_count == 1
    assert assessment.center_inside
    assert not assessment.interlocked


def test_support_polygon_accepts_two_independent_contacts_around_centroid() -> None:
    assessment = _support_polygon_assessment(
        target=(0.50, 0.50, 0.40, 0.40),
        supports=[
            (0.40, 0.50, 0.20, 0.40),
            (0.60, 0.50, 0.20, 0.40),
        ],
        min_contact_ratio=0.01,
        min_hull_area_ratio=0.01,
        center_margin_ratio=0.0,
    )

    assert assessment.contact_count == 2
    assert not assessment.degenerate
    assert assessment.center_inside
    assert assessment.interlocked


def test_support_polygon_rejects_overlapping_lower_boxes_that_duplicate_area() -> None:
    assessment = _support_polygon_assessment(
        target=(0.50, 0.50, 0.40, 0.40),
        supports=[
            (0.50, 0.50, 0.40, 0.40),
            (0.65, 0.50, 0.40, 0.40),
        ],
        min_contact_ratio=0.01,
        min_hull_area_ratio=0.01,
        center_margin_ratio=0.0,
    )

    # 100% + 62.5% de contacto sobre la misma caja superior viola la
    # disjunción esencial del nivel inferior; el hull no puede contar área
    # duplicada como dos apoyos físicos independientes.
    assert sum(assessment.shares) > 1.0
    assert not assessment.interlocked


def test_dynamic_k_fallback_accepts_four_balanced_supports() -> None:
    cfg = load_palletizing_config(
        Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    )
    counter = GridCounter(
        np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32),
        cfg,
    )

    assert counter._dynamic_support_is_balanced(
        [0.26, 0.25, 0.24, 0.23], min_coverage=0.75, max_ratio=2.0,
    )
    assert not counter._dynamic_support_is_balanced(
        [0.70, 0.10, 0.10, 0.08], min_coverage=0.75, max_ratio=2.0,
    )


def test_square_consensus_suppresses_unobservable_orientation() -> None:
    cfg = load_palletizing_config(
        Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    )
    counter = GridCounter(
        np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32),
        cfg,
    )
    counter._footprint[(0, 0)] = (0.399, 0.401)
    counter._footprint[(1, 0)] = (0.401, 0.399)
    counter._level_footprint[0] = (0.401, 0.399)

    # La diferencia de lados es menor que la incertidumbre geométrica: no
    # existe evidencia física para conservar dos orientaciones distintas.
    assert counter._canonical_footprint((0, 0)) == (0.401, 0.399)
    assert counter._canonical_footprint((1, 0)) == (0.401, 0.399)


def test_repeated_candidate_diagnostic_is_recorded_once_per_arm_cycle() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    cfg = load_palletizing_config(config_path)
    counter = GridCounter(
        np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32),
        cfg,
    )

    counter._log_new_candidate(0.246, 0.568, (0.107, 0.162))
    counter._log_new_candidate(0.246, 0.568, (0.107, 0.162))
    assert len(counter._logged_candidate_diagnostics) == 1

    counter._close_arm_cycle()
    assert not counter._logged_candidate_diagnostics


def test_coin_roll_10_productive_capacity_is_25() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    cfg = load_palletizing_config(config_path)

    assert cfg.boxes_per_level["coin_roll_10"] == 25


def test_config_has_no_class_specific_support_thresholds() -> None:
    source = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg = load_palletizing_config(source)

    assert "class_thresholds" not in raw
    assert not hasattr(cfg, "class_thresholds")
    assert not hasattr(cfg, "support_thresholds_for")


def _gate(**overrides) -> SimpleNamespace:
    """Gate que no estorba: sin pausa por movimiento, para que los tests de
    conteo no tengan que simular frames quietos. Los tests del gate lo
    sobreescriben con lo que necesitan."""
    base = dict(
        motion_pause_enabled=False,
        motion_diff_threshold=6.0,
        motion_stable_frames=2,
        arm_debounce_frames=3,
        # Deliberadamente enorme: los tests de conteo existentes no piensan
        # en terminos de "cuantos frames vacios seguidos", asi que un
        # default chico les dispararia un reset de paleta que no esperan.
        # Los tests que SI prueban ese reset lo overridean con un numero chico.
        empty_pallet_debounce_frames=10_000,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _still(boxes) -> FrameInput:
    """Frame sin brazo y sin movimiento: la escena en reposo."""
    return FrameInput(boxes=boxes, arm_visible=False, motion_score=0.0)


class PalletizingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SimpleNamespace(
            reference_scale_px=20.0,
            c_z=3.0,
            box_height=0.3,
            levels_layout=(SimpleNamespace(cells=((0.5, 0.5),)),),
            levels=1,
            layout_mode="auto",
            boxes_per_level={},
            occupancy_grid=200,
            confirmation=SimpleNamespace(min_stable=1, same_box_iou=0.25),
            gate=_gate(),
            tau_rung=0.20,
            tau_rec=0.12,
            tau_cell=0.08,
            tau_overlap=0.40,
            tau_overlap_center=0.60,
            tau_cell_overlap=0.35,
            min_stack_area_ratio=0.80,
            max_duplicate_scale_ratio=0.85,
            free_gap_ratio=0.85,
            min_support_coverage=0.75,
            max_support_ratio=2.0,
            # Preexistente: estos 5 campos existen en PalletizingConfig real
            # (configs/palletizing.yaml) pero faltaban en este fixture --
            # cualquier test que confirmara una caja (count_changed=True)
            # reventaba en _log_overlapping_cells() con AttributeError sobre
            # overlap_warn_ratio. Valores iguales a los defaults de
            # configs/palletizing.yaml.
            overlap_warn_ratio=0.15,
            max_same_level_overlap=0.10,
            min_complete_side_ratio=0.80,
            partial_fit_tolerance=0.028,
            max_bootstrap_combinations=20_000,
        )
        self.roi = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)

    def two_level_cfg(self) -> SimpleNamespace:
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.levels_layout = (
            SimpleNamespace(cells=((0.5, 0.5),)),
            SimpleNamespace(cells=((0.5, 0.5),)),
        )
        cfg.levels = 2
        cfg.reference_scale_px = 200.0
        return cfg


class AutoLayoutTests(PalletizingTestCase):
    def _confirmed_vertical_cell(self) -> GridCounter:
        counter = GridCounter(self.roi, self.cfg)
        key = (0, 0)
        counter._occupied.add(key)
        counter._dynamic_positions[key] = (0.50, 0.50)
        counter._footprint[key] = (0.10, 0.30)
        counter._level_footprint[0] = (0.30, 0.10)
        return counter

    def test_crossed_horizontal_box_does_not_match_vertical_identity(self) -> None:
        counter = self._confirmed_vertical_cell()
        # Interseccion 0.10x0.10 sobre deteccion 0.25x0.10: overlap/min=40%,
        # supera tau=35%, pero solo 40% de la DETECCION queda contenido.
        parsed = [((38, 45, 63, 55), 0.95, "test_box")]

        self.assertEqual({}, counter._match_to_cells(parsed))

    def test_contained_occlusion_fragment_matches_confirmed_identity(self) -> None:
        counter = self._confirmed_vertical_cell()
        # Fragmento 0.08x0.20 completamente dentro de la vertical 0.10x0.30.
        parsed = [((46, 40, 54, 60), 0.95, "test_box")]

        self.assertEqual({0: (0, 0)}, counter._match_to_cells(parsed))

    def test_extra_contained_fragment_only_validates_used_identity(self) -> None:
        counter = self._confirmed_vertical_cell()
        parsed = [
            ((46, 36, 54, 64), 0.95, "test_box"),  # observacion principal
            ((46, 40, 54, 60), 0.44, "test_box"),  # pedazo visible inferior
        ]
        matched = counter._match_to_cells(parsed)

        self.assertEqual({0: (0, 0)}, matched)
        self.assertEqual(
            {1: (0, 0)},
            counter._contained_validation_fragments(parsed, matched),
        )

    def test_larger_detection_cannot_become_validation_fragment(self) -> None:
        counter = self._confirmed_vertical_cell()
        parsed = [
            ((45, 35, 55, 65), 0.95, "test_box"),
            ((44, 30, 56, 70), 0.44, "test_box"),  # crece: no es recorte
        ]
        matched = {0: (0, 0)}

        self.assertEqual({}, counter._contained_validation_fragments(parsed, matched))

    def test_new_box_persists_temporal_median_not_last_bbox(self) -> None:
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.confirmation = SimpleNamespace(min_stable=4, same_box_iou=0.25)
        counter = GridCounter(self.roi, cfg)

        sequence = [
            (10, 30, 40, 70, 0.91),  # primer frame todavía desplazado
            (20, 30, 50, 70, 0.93),
            (20, 30, 50, 70, 0.95),
            (30, 30, 60, 70, 0.92),  # último frame tampoco manda por sí solo
        ]
        for detection in sequence:
            result = counter.update(_still([detection]))

        self.assertEqual(CellState.NEW, result.detections[0].state)
        box = counter.scene_state(0.6).boxes[0]
        self.assertAlmostEqual(0.35, box.u, places=2)
        self.assertAlmostEqual(0.50, box.v, places=2)

    def test_partial_validation_is_not_drawn(self) -> None:
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        before = frame.copy()

        draw_grid_detections(frame, [
            GridDetection(
                (5, 5, 30, 30), 0, 0, CellState.VALIDATION,
                "valida-oclusion-superior", 0.95,
            ),
        ], SimpleNamespace())

        self.assertTrue(np.array_equal(before, frame))

    def test_new_box_requires_configured_consecutive_observations(self) -> None:
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.confirmation = SimpleNamespace(min_stable=3, same_box_iou=0.25)
        counter = GridCounter(self.roi, cfg)

        first = counter.update(_still([(10, 40, 30, 60, 0.95)]))
        second = counter.update(_still([(11, 40, 31, 60, 0.94)]))
        third = counter.update(_still([(12, 40, 32, 60, 0.93)]))

        self.assertIsNone(first.detections[0].cell)
        self.assertEqual(CellState.REJECTED, first.detections[0].state)
        self.assertEqual(CellState.REJECTED, second.detections[0].state)
        self.assertEqual(CellState.NEW, third.detections[0].state)
        self.assertEqual(1, counter.total)

    def test_first_observation_is_available_as_confirming_iso_geometry(self) -> None:
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.confirmation = SimpleNamespace(min_stable=3, same_box_iou=0.25)
        counter = GridCounter(self.roi, cfg)

        preview = counter.provisional_boxes(
            [(10, 40, 30, 60, 0.95, "coin_roll_100")],
            height_ratio=0.6,
            level_tops=[0.0],
        )

        self.assertEqual(0, counter.total, "mostrar no debe confirmar ni contar")
        self.assertEqual(1, len(preview))
        self.assertEqual("confirming", preview[0].status)
        self.assertEqual(-1, preview[0].cell)
        self.assertEqual(0, preview[0].level)

    def test_partial_observation_cannot_preview_as_a_new_level(self) -> None:
        counter = self._confirmed_vertical_cell()
        counter._tracking_floor_level = 0
        counter._footprint.update({(1, 1): (0.30, 0.10), (2, 1): (0.30, 0.10), (3, 1): (0.30, 0.10)})
        counter._level_footprint[1] = (0.30, 0.10)
        counter._match_to_cells = lambda _parsed: {0: (0, 0)}
        counter._is_stack_candidate = lambda *_args: True

        preview = counter.provisional_boxes(
            [(40, 40, 50, 60, 0.95, "test_box")],
            height_ratio=0.6,
            level_tops=[0.0, 0.12],
        )

        assert preview == []

    def test_initial_bootstrap_exposes_internal_geometry_as_initializing(self) -> None:
        counter = GridCounter(self.roi, self.cfg)
        counter._occupied.add((0, 0))
        counter._dynamic_positions[(0, 0)] = (0.50, 0.50)
        counter._footprint[(0, 0)] = (0.20, 0.20)
        counter._level_footprint[0] = (0.20, 0.20)
        counter._initial_scene_deferred = True

        scene = counter.scene_state(0.6)

        self.assertEqual([], scene.boxes)
        self.assertTrue(scene.validating_initial)
        self.assertEqual(1, len(scene.provisional_boxes))
        self.assertEqual("initializing", scene.provisional_boxes[0].status)

    def test_auto_layout_is_validating_from_first_frame(self) -> None:
        counter = GridCounter(self.roi, self.cfg)

        scene = counter.scene_state(0.6)

        self.assertTrue(scene.validating_initial)

    def test_bootstrap_preview_includes_matched_observations_as_initializing(self) -> None:
        counter = self._confirmed_vertical_cell()

        preview = counter.provisional_boxes(
            [(40, 40, 60, 60, 0.95, "coin_roll_100")],
            height_ratio=0.6,
            level_tops=[0.0, 0.12],
        )

        self.assertEqual(1, len(preview))
        self.assertEqual("initializing", preview[0].status)

    def test_disappearing_candidate_restarts_validation_from_zero(self) -> None:
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.confirmation = SimpleNamespace(min_stable=3, same_box_iou=0.25)
        counter = GridCounter(self.roi, cfg)
        box = (10, 40, 30, 60, 0.95)

        counter.update(_still([box]))
        counter.update(_still([]))  # desapareció: la primera evidencia no vale
        restarted_1 = counter.update(_still([box]))
        restarted_2 = counter.update(_still([box]))
        restarted_3 = counter.update(_still([box]))

        self.assertEqual(CellState.REJECTED, restarted_1.detections[0].state)
        self.assertEqual(CellState.REJECTED, restarted_2.detections[0].state)
        self.assertEqual(CellState.NEW, restarted_3.detections[0].state)

    def test_discovers_every_distinct_box_instead_of_forcing_sample_cells(self) -> None:
        counter = GridCounter(self.roi, self.cfg)

        result = counter.update(_still([
            (10, 40, 30, 60, 0.95),
            (40, 40, 60, 60, 0.94),
            (70, 40, 90, 60, 0.93),
        ]))

        self.assertEqual(GateState.COUNTING, result.gate)
        self.assertTrue(result.count_changed)
        self.assertEqual(3, counter.visible)
        self.assertEqual(3, counter.total)
        self.assertEqual([CellState.NEW] * 3, [d.state for d in result.detections])
        scene = counter.scene_state(0.6)
        self.assertEqual([(0, 0), (1, 0), (2, 0)],
                         sorted((box.cell, box.level) for box in scene.boxes))

    def test_measured_non_overlapping_neighbour_gets_a_distinct_cell(self) -> None:
        cfg = SimpleNamespace(**vars(self.cfg))
        # Caso real: usar el lado maximo (0.173) como tau_cell seria incluso
        # mas permisivo que el 0.12 actual frente a centros separados 0.10.
        cfg.tau_cell = 0.173
        counter = GridCounter(self.roi, cfg)
        counter._occupied.add((0, 0))
        counter._dynamic_positions[(0, 0)] = (0.30, 0.50)
        counter._footprint[(0, 0)] = (0.10, 0.173)
        counter._next_cell_by_level[0] = 1

        cell, reason = counter._assign_cell(40.0, 50.0, 0, (0.10, 0.173))

        self.assertEqual("", reason)
        self.assertEqual(1, cell)
        # `_assign_cell` propone identidad; solo la transicion NEW confirmada
        # puede persistir posicion e incrementar el siguiente id.
        self.assertNotIn((1, 0), counter._dynamic_positions)
        self.assertEqual(1, counter._next_cell_by_level[0])

    def test_matches_jittered_redetection_to_discovered_cell(self) -> None:
        counter = GridCounter(self.roi, self.cfg)
        counter.update(_still([(10, 40, 30, 60, 0.95)]))

        result = counter.update(_still([(12, 41, 32, 61, 0.92)]))

        self.assertEqual(CellState.REDET, result.detections[0].state)
        self.assertFalse(result.count_changed)
        self.assertEqual(1, counter.total)
        self.assertAlmostEqual(0.20, counter.scene_state(0.6).boxes[0].u, places=2)

    def test_spatial_identity_preserves_box_when_bbox_becomes_partial(self) -> None:
        counter = GridCounter(self.roi, self.two_level_cfg())
        # Esta prueba cubre tracking posterior, no la vista provisional del
        # bootstrap inicial.
        counter._bootstrap_reconciled = True
        first = counter.update(_still([(-7, -7, 108, 108, 0.90)])).detections[0]
        before = counter.scene_state(0.6).boxes[0]

        partial = None
        for _ in range(3):
            partial = counter.update(_still([(20, 20, 70, 70, 0.60)])).detections[0]
        after = counter.scene_state(0.6).boxes[0]

        self.assertIsNotNone(partial)
        self.assertEqual(CellState.NEW, first.state)
        self.assertEqual(CellState.REDET, partial.state)
        self.assertEqual(first.level, partial.level)
        self.assertEqual(first.cell, partial.cell)
        self.assertEqual((before.side_a, before.side_b), (after.side_a, after.side_b))
        self.assertEqual(1, counter.total)

    def test_non_contained_detection_cannot_move_confirmed_iso_box(self) -> None:
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.confirmation = SimpleNamespace(min_stable=3, same_box_iou=0.25)
        counter = GridCounter(self.roi, cfg)
        for _ in range(3):
            counter.update(_still([(20, 20, 80, 80, 0.95)]))
        before = counter.scene_state(0.6).boxes[0]

        for _ in range(3):
            counter.update(_still([(40, 20, 100, 80, 0.96)]))
        after = counter.scene_state(0.6).boxes[0]

        self.assertAlmostEqual(0.50, after.u, places=2)
        self.assertEqual(before.level, after.level)
        self.assertEqual((before.side_a, before.side_b), (after.side_a, after.side_b))

    def test_contained_redetections_never_move_confirmed_position(self) -> None:
        counter = GridCounter(self.roi, self.cfg)
        key = (0, 0)
        counter._occupied.add(key)
        counter._dynamic_positions[key] = (0.50, 0.50)
        counter._footprint[key] = (0.20, 0.20)
        counter._level_footprint[0] = (0.20, 0.20)
        counter._bootstrap_reconciled = True

        parsed = [((43, 40, 63, 60), 0.95, "test_box")]
        for _ in range(3):
            self.assertEqual({0: key}, counter._match_to_cells(parsed))

        self.assertEqual((0.50, 0.50), counter._dynamic_positions[key])

    def test_confirmed_identity_never_changes_orientation_from_crossed_detection(self) -> None:
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.confirmation = SimpleNamespace(min_stable=3, same_box_iou=0.25)
        cfg.max_position_correction = 0.04
        counter = GridCounter(self.roi, cfg)
        key = (3, 0)
        counter._occupied.add(key)
        counter._dynamic_positions[key] = (0.74, 0.71)
        counter._footprint[key] = (0.482, 0.190)  # observacion inicial inflada y horizontal
        counter._level_footprint[0] = (0.305, 0.160)
        counter._bootstrap_reconciled = True

        crossed = [((66, 56, 82, 86), 0.95, "test_box")]
        self.assertEqual({}, counter._match_to_cells(crossed))

        unchanged = counter.scene_state(0.6).boxes[0]
        self.assertAlmostEqual(0.74, unchanged.u, places=2)
        self.assertAlmostEqual(0.71, unchanged.v, places=2)
        self.assertGreater(unchanged.side_a, unchanged.side_b)
        self.assertEqual((0.482, 0.190), counter._footprint[key])

    def test_inflated_raw_footprint_cannot_steal_a_new_upper_box(self) -> None:
        counter = GridCounter(self.roi, self.two_level_cfg())
        key = (3, 1)
        counter._occupied.add(key)
        counter._dynamic_positions[key] = (0.741, 0.713)
        counter._footprint[key] = (0.482, 0.190)  # bbox inicial anómalo del log real
        counter._level_footprint[1] = (0.300, 0.160)  # tamaño canónico mostrado por ISO

        parsed = [((79, 63, 96, 93), 0.95)]  # candidata real en (0.875, 0.780)

        self.assertEqual({}, counter._match_to_cells(parsed))
        cell, _reason = counter._assign_cell(
            87.5, 78.0, 1, (0.17, 0.30), reuse_occupied=False,
        )
        self.assertNotEqual(3, cell)
        self.assertAlmostEqual(0.875, counter._cell_position(cell, 1)[0], places=2)

    def test_interlocked_box_is_promoted_without_temporal_identity(self) -> None:
        cfg = self.two_level_cfg()
        cfg.boxes_per_level = {"test_box": 2}
        cfg.min_supporting_boxes = 2
        cfg.min_support_coverage = 0.75
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")
        counter.update(_still([
            (10, 30, 50, 70, 0.95),
            (50, 30, 90, 70, 0.94),
        ]))
        counter.update(FrameInput([], arm_visible=True, motion_score=0.0))
        for _ in range(cfg.gate.arm_debounce_frames):
            counter.update(_still([]))

        result = counter.update(_still([(30, 30, 70, 70, 0.96)]))

        self.assertEqual(CellState.NEW, result.detections[0].state)
        self.assertEqual(1, result.detections[0].level)
        self.assertEqual(3, counter.total)

    def test_box_over_one_support_is_only_a_redetection(self) -> None:
        cfg = self.two_level_cfg()
        cfg.boxes_per_level = {"test_box": 1}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")
        counter.update(_still([(30, 30, 70, 70, 0.95)]))
        counter.update(FrameInput([], arm_visible=True, motion_score=0.0))
        for _ in range(cfg.gate.arm_debounce_frames):
            counter.update(_still([]))

        result = counter.update(_still([(30, 30, 70, 70, 0.96)]))

        self.assertEqual(CellState.REDET, result.detections[0].state)
        self.assertEqual(0, result.detections[0].level)
        self.assertEqual(1, counter.total)

    def test_almost_exactly_over_one_box_is_not_interlocked_by_tiny_second_support(self) -> None:
        cfg = self.two_level_cfg()
        cfg.boxes_per_level = {"test_box": 2}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")
        counter._occupied.update({(0, 0), (1, 0)})
        counter._dynamic_positions.update({(0, 0): (0.50, 0.50), (1, 0): (0.65, 0.50)})
        counter._footprint.update({(0, 0): (0.40, 0.40), (1, 0): (0.20, 0.20)})
        counter._level_footprint[0] = (0.40, 0.40)
        counter._current_frame = 10

        level = counter._stacking_level(0.50, 0.50, (0.40, 0.40))

        self.assertIsNone(level)

    def test_two_supports_do_not_promote_until_lower_level_is_full(self) -> None:
        cfg = self.two_level_cfg()
        cfg.boxes_per_level = {"test_box": 3}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")
        counter.update(_still([
            (10, 30, 50, 70, 0.95),
            (50, 30, 90, 70, 0.94),
        ]))
        counter.update(FrameInput([], arm_visible=True, motion_score=0.0))
        for _ in range(cfg.gate.arm_debounce_frames):
            counter.update(_still([]))

        result = counter.update(_still([(30, 30, 70, 70, 0.96)]))

        self.assertEqual(CellState.REDET, result.detections[0].state)
        self.assertEqual(0, result.detections[0].level)
        self.assertEqual(2, counter.total)

    def test_level_is_full_only_at_exact_capacity(self) -> None:
        cfg = self.two_level_cfg()
        cfg.boxes_per_level = {"test_box": 2}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")
        counter._arm_cycle_seen = True  # capacidad exacta solo aplica en operacion, no en bootstrap
        counter._occupied.update({(0, 0), (1, 0)})

        self.assertTrue(counter._level_is_full(0))
        counter._occupied.add((2, 0))
        self.assertFalse(counter._level_is_full(0))

    def test_bootstrap_full_falls_back_to_geometry_when_capacity_unreachable(self) -> None:
        """Antes del primer ciclo de brazo, una caja del piso tapada desde el
        primer frame nunca entra a `_occupied` -- exigir la capacidad exacta
        (3) la dejaria sin promover para siempre. El chequeo geometrico de
        huecos libres SI puede darse por lleno con lo que alcanzo a ver."""
        cfg = self.two_level_cfg()
        cfg.boxes_per_level = {"test_box": 3}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")
        self.assertFalse(counter.arm_cycle_seen)  # todavia en inventario inicial

        # Solo 2 de las 3 cajas del nivel 0 fueron visibles alguna vez; la
        # tercera nacio tapada. Su footprint cubre TODO el nivel 0 [0,1]x[0,1]
        # entre las dos visibles -- no queda hueco donde meter otra.
        counter._occupied.update({(0, 0), (1, 0)})
        counter._dynamic_positions.update({(0, 0): (0.25, 0.5), (1, 0): (0.75, 0.5)})
        counter._footprint.update({(0, 0): (0.5, 1.0), (1, 0): (0.5, 1.0)})
        counter._level_footprint[0] = (1.0, 0.5)

        self.assertTrue(counter._level_is_full(0))

    def test_bootstrap_finishes_single_level_even_if_arm_cycle_was_seen(self) -> None:
        """El brazo puede estar visible durante el arranque. Ese hecho no
        puede abandonar un inventario inicial todavia sin reconciliar: una
        escena estable sin parciales debe cerrar bootstrap y publicar el ISO.
        """
        cfg = self.two_level_cfg()
        cfg.boxes_per_level = {"test_box": 2}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")
        counter._occupied.add((0, 0))
        counter._dynamic_positions[(0, 0)] = (0.50, 0.50)
        counter._footprint[(0, 0)] = (0.30, 0.20)
        counter._level_footprint[0] = (0.30, 0.20)
        counter._arm_cycle_seen = True

        counter._reconcile_initial_layers([
            ((35, 40, 65, 60), 0.95, "test_box"),
        ])

        self.assertTrue(counter._bootstrap_reconciled)
        self.assertFalse(counter._initial_scene_deferred)
        self.assertEqual(1, len(counter.scene_state(0.6).boxes))

    def test_box_above_covers_the_hole_of_the_box_it_hides(self) -> None:
        """Invariante de area: una caja del nivel 1 tapa exactamente un
        footprint del nivel 0. Ese hueco del raster no es espacio libre --
        es la caja que la camara nunca vio. Sin contarlo, una paleta que
        arranca apilada nunca da el nivel 0 por lleno (deadlock)."""
        cfg = self.two_level_cfg()
        cfg.boxes_per_level = {"test_box": 3}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")

        # Nivel 0 de 3 celdas en columna; la del medio nacio tapada y nunca
        # entro a `_occupied`. Solo se confirmaron la de arriba y la de abajo.
        counter._occupied.update({(0, 0), (1, 0)})
        counter._dynamic_positions.update({(0, 0): (0.5, 1 / 6), (1, 0): (0.5, 5 / 6)})
        counter._footprint.update({(0, 0): (1.0, 1 / 3), (1, 0): (1.0, 1 / 3)})
        counter._level_footprint[0] = (1.0, 1 / 3)

        # Con el hueco del medio a la vista, el nivel NO esta lleno.
        self.assertFalse(counter._level_is_full(0))

        # La caja del nivel 1 ocupa justo ese hueco: ya no queda nada libre.
        counter._occupied.add((0, 1))
        counter._dynamic_positions[(0, 1)] = (0.5, 0.5)
        counter._footprint[(0, 1)] = (1.0, 1 / 3)
        counter._level_footprint[1] = (1.0, 1 / 3)

        self.assertTrue(counter._level_is_full(0))

    def test_occlusion_never_stacks_a_box_exactly_over_a_single_one(self) -> None:
        """Invariante fisico: una caja del nivel i+1 esta TRABADA sobre el
        nivel i -- cruza dos o mas cajas, nunca queda alineada con una sola.
        Una caja exactamente encima de otra es indistinguible de una
        re-deteccion desde una cenital, asi que la ruta de oclusion no puede
        promoverla."""
        cfg = self.two_level_cfg()
        cfg.boxes_per_level = {"test_box": 1}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")
        counter._arm_cycle_seen = True

        # Una unica celda confirmada en el nivel 0: el nivel esta lleno.
        counter._occupied.add((0, 0))
        counter._dynamic_positions[(0, 0)] = (0.5, 0.5)
        counter._footprint[(0, 0)] = (0.4, 0.4)
        counter._level_footprint[0] = (0.4, 0.4)

        # Par de oclusion en la MISMA posicion: un recorte y una caja
        # completa, ambos centrados sobre la celda (0, 0).
        result = counter.update(_still([
            (30, 30, 70, 70, 0.95),   # completa, encima de (0,0)
            (34, 34, 58, 58, 0.90),   # recorte solapado, mas chico
        ]))

        promoted = [d for d in result.detections if d.level == 1]
        self.assertEqual([], promoted)
        self.assertNotIn((0, 1), counter._occupied)

    def test_partial_box_marks_lower_level_proven_full(self) -> None:
        """Encadenamiento fisico: hay un recorte -> algo lo tapa -> eso esta
        trabado encima -> es del nivel i -> el nivel i-1 no podia tener hueco.
        El par de oclusion deja constancia de esa prueba."""
        counter = self._interlocked_counter()
        counter._arm_cycle_seen = False  # inventario inicial
        self.assertEqual(set(), counter._proven_full)

        counter.update(_still([
            self._UPPER_BBOX + (0.70,),
            self._PARTIAL_BBOX + (0.98,),
        ]))

        self.assertIn(0, counter._proven_full)

    def test_proven_full_beats_capacity_and_geometry_only_in_bootstrap(self) -> None:
        """Un nivel probado lleno por oclusion vale aunque sus cajas tapadas
        nunca hayan llegado a `_occupied` -- ni la cuenta exacta ni el raster
        de huecos pueden verlas. Cerrado el primer ciclo manda la capacidad,
        que ya es alcanzable porque cada caja nueva se observa sin oclusion."""
        cfg = self.two_level_cfg()
        cfg.boxes_per_level = {"test_box": 3}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")
        # Una sola celda chica: ni la capacidad (1 de 3) ni la geometria
        # (sobra lugar de sobra) darian el nivel por lleno.
        counter._occupied.add((0, 0))
        counter._dynamic_positions[(0, 0)] = (0.2, 0.2)
        counter._footprint[(0, 0)] = (0.2, 0.2)
        counter._level_footprint[0] = (0.2, 0.2)
        self.assertFalse(counter._level_is_full(0))

        counter._proven_full.add(0)
        self.assertTrue(counter._level_is_full(0))

        counter._arm_cycle_seen = True
        self.assertFalse(counter._level_is_full(0))

    def test_partial_grows_toward_the_side_without_a_neighbour(self) -> None:
        """Un recorte no dice donde termina la caja, pero el tamano canonico
        si dice cuanto mide. Entre las dos direcciones posibles manda la
        fisica: crece hacia el lado libre, nunca encima de una vecina."""
        cfg = self.two_level_cfg()
        cfg.boxes_per_level = {"test_box": 3}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")
        # Vecina confirmada pegada a la izquierda: u 0.0..0.3
        counter._occupied.add((0, 0))
        counter._dynamic_positions[(0, 0)] = (0.15, 0.5)
        counter._footprint[(0, 0)] = (0.3, 0.6)
        counter._level_footprint[0] = (0.6, 0.3)

        # Fragmento angosto pegado al borde derecho de la vecina. La caja
        # completa mide 0.3 de ancho: o crece a la izquierda (encima de la
        # vecina) o a la derecha (libre). Debe elegir la derecha.
        completion = counter._complete_partial_footprint(0.35, 0.5, (0.1, 0.6), 0)

        self.assertIsNotNone(completion)
        (cu, cv), full = completion
        self.assertEqual((0.3, 0.6), full)
        # Crecio a la derecha: su borde izquierdo no invade a la vecina, que
        # termina en u=0.30. Crecer a la izquierda la habria puesto en 0.15.
        self.assertGreaterEqual(cu - full[0] / 2.0, 0.30 - 1e-6)
        self.assertLessEqual(counter._max_same_level_overlap(cu, cv, full, 0), 0.10)

    def test_partial_is_not_completed_when_both_directions_are_free(self) -> None:
        """Sin vecinas que desempaten, las dos direcciones son igual de
        validas: no se deduce nada antes que inventar una posicion."""
        cfg = self.two_level_cfg()
        cfg.boxes_per_level = {"test_box": 3}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")
        # Ninguna celda confirmada en el nivel: nada que desempate hacia que
        # lado crece el fragmento.
        counter._level_footprint[0] = (0.6, 0.3)

        self.assertIsNone(
            counter._complete_partial_footprint(0.4, 0.5, (0.1, 0.6), 0)
        )

    def test_capacity_rejects_an_extra_box_in_the_same_level(self) -> None:
        cfg = self.cfg
        cfg.boxes_per_level = {"test_box": 1}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")

        first = counter.update(_still([(10, 30, 40, 70, 0.95)]))
        second = counter.update(_still([(60, 30, 90, 70, 0.96)]))

        self.assertEqual(CellState.NEW, first.detections[0].state)
        self.assertEqual(CellState.REJECTED, second.detections[0].state)
        self.assertEqual("nivel-lleno", second.detections[0].reason)
        self.assertEqual(1, counter.total)

    def test_rejected_tentative_cells_do_not_consume_permanent_ids(self) -> None:
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.boxes_per_level = {"test_box": 1}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")
        counter.update(_still([(5, 40, 25, 60, 0.95)]))
        self.assertEqual(1, counter._next_cell_by_level[0])

        for _ in range(5):
            rejected = counter.update(_still([(70, 40, 90, 60, 0.96)]))
            self.assertEqual(CellState.REJECTED, rejected.detections[0].state)

        self.assertEqual(1, counter._next_cell_by_level[0])
        self.assertEqual({(0, 0)}, set(counter._dynamic_positions))

    def test_capacity_25_keeps_box_25_on_level_zero_after_bootstrap(self) -> None:
        cfg = self.two_level_cfg()
        cfg.reference_scale_px = 10.0
        cfg.boxes_per_level = {"test_box": 25}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")
        counter._bootstrap_reconciled = True

        positions = [
            (u, v)
            for v in (0.10, 0.30, 0.50, 0.70, 0.90)
            for u in (0.10, 0.30, 0.50, 0.70, 0.90)
        ]
        for cell, position in enumerate(positions[:24]):
            key = (cell, 0)
            counter._occupied.add(key)
            counter._dynamic_positions[key] = position
            counter._footprint[key] = (0.10, 0.10)
            counter._cell_frame[key] = 1
        counter._level_footprint[0] = (0.10, 0.10)
        counter._next_cell_by_level[0] = 24
        counter.total = counter.initial = 24
        counter._arm_cycle_seen = True
        counter._placement_credits = 1

        self.assertFalse(counter._level_is_full(0))
        result = counter.update(_still([(85, 85, 95, 95, 0.96)]))

        self.assertEqual(CellState.NEW, result.detections[0].state)
        self.assertEqual(0, result.detections[0].level)
        self.assertIn((24, 0), counter._occupied)
        self.assertTrue(counter._level_is_full(0))

    def test_level_footprint_uses_median_of_short_and_long_sides(self) -> None:
        counter = GridCounter(self.roi, self.cfg)
        counter._footprint.update({
            (0, 0): (0.40, 0.20),
            (1, 0): (0.22, 0.42),
            (2, 0): (0.90, 0.90),  # outlier inflado
        })

        counter._recompute_level_footprint(0)

        self.assertEqual((0.42, 0.22), counter._level_footprint[0])

    def test_partial_unmatched_box_cannot_become_new(self) -> None:
        counter = GridCounter(self.roi, self.cfg)
        counter.update(_still([
            (5, 10, 25, 70, 0.95),
            (40, 10, 60, 70, 0.94),
            (75, 10, 95, 70, 0.93),
        ]))

        partial = counter.update(_still([(10, 80, 20, 95, 0.94)]))

        self.assertEqual(CellState.REJECTED, partial.detections[0].state)
        self.assertEqual("recorte", partial.detections[0].reason)
        self.assertEqual(3, counter.total)

    def test_same_level_overlap_is_rejected_before_confirmation(self) -> None:
        counter = GridCounter(self.roi, self.cfg)
        counter.update(_still([(10, 30, 40, 70, 0.95)]))

        overlapping = counter.update(_still([(35, 30, 65, 70, 0.94)]))

        self.assertEqual(CellState.REJECTED, overlapping.detections[0].state)
        self.assertEqual("solape-intranivel", overlapping.detections[0].reason)
        self.assertEqual(1, counter.total)

    # Geometria trabada, la unica que existe en este proyecto: dos celdas del
    # nivel 0 lado a lado y una del nivel 1 a caballo entre las dos. Una caja
    # exactamente encima de UNA sola es indistinguible de una re-deteccion
    # desde una cenital, asi que no sirve como andamio de estos tests.
    #
    #   nivel 0:  [ (0,0) ][ (1,0) ]      nivel 1:      [ (0,1) ]
    #             0      50      100                   25      75
    _UPPER_BBOX = (25, 25, 75, 75)    # completa, trabada sobre las dos de abajo
    _PARTIAL_BBOX = (15, 30, 50, 65)  # parte visible de (0,0), recortada por la de arriba

    def _interlocked_counter(self) -> GridCounter:
        """Nivel 0 lleno con dos celdas lado a lado y un ciclo de brazo ya
        cerrado: el punto de partida para evaluar una caja de nivel 1."""
        cfg = self.two_level_cfg()
        cfg.boxes_per_level = {"test_box": 2}
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("test_box")
        counter._arm_cycle_seen = True
        counter._placement_credits = 1
        counter._occupied.update({(0, 0), (1, 0)})
        counter._dynamic_positions.update({(0, 0): (0.25, 0.5), (1, 0): (0.75, 0.5)})
        counter._footprint.update({(0, 0): (0.5, 0.5), (1, 0): (0.5, 0.5)})
        counter._level_footprint[0] = (0.5, 0.5)
        counter._cell_frame.update({(0, 0): 1, (1, 0): 1})
        counter._next_cell_by_level[0] = 2
        counter.total = counter.initial = 2
        counter._current_frame = 5
        counter._bootstrap_reconciled = True
        return counter

    def _stack_upper_cell(self, counter: GridCounter) -> None:
        """Agrega la caja del nivel 1 ya confirmada, trabada sobre las dos."""
        counter._occupied.add((0, 1))
        counter._dynamic_positions[(0, 1)] = (0.5, 0.5)
        counter._footprint[(0, 1)] = (0.5, 0.5)
        counter._level_footprint[1] = (0.5, 0.5)
        counter._cell_frame[(0, 1)] = 2
        counter._next_cell_by_level[1] = 1
        counter.total = 3
        counter.initial = 3

    def test_partial_lower_bbox_validates_upper_box_without_becoming_new(self) -> None:
        """La parcial confirma la relacion i -> i+1 sin crear ni deformar
        nada: es VALIDATION, jamas NEW."""
        counter = self._interlocked_counter()
        lower_before = next(b for b in counter.scene_state(0.6).boxes if b.cell == 0)

        result = counter.update(_still([
            self._UPPER_BBOX + (0.70,),    # caja completa superior, trabada
            self._PARTIAL_BBOX + (0.98,),  # parte visible de la inferior
        ]))

        by_bbox = {d.bbox: d for d in result.detections}
        upper = by_bbox[self._UPPER_BBOX]
        partial = by_bbox[self._PARTIAL_BBOX]
        lower_after = next(b for b in counter.scene_state(0.6).boxes if b.cell == 0 and b.level == 0)
        self.assertEqual(3, counter.total)
        self.assertEqual(CellState.NEW, upper.state)
        self.assertEqual(1, upper.level)
        self.assertEqual(CellState.VALIDATION, partial.state)
        self.assertEqual(0, partial.level)
        self.assertEqual(
            (lower_before.side_a, lower_before.side_b),
            (lower_after.side_a, lower_after.side_b),
        )

    def test_redetection_over_stacked_cells_keeps_highest_confirmed_level(self) -> None:
        """La identidad espacial conserva la caja superior aunque varias
        celdas apiladas compartan la misma zona de imagen."""
        counter = self._interlocked_counter()
        self._stack_upper_cell(counter)

        redetected = counter.update(_still([self._UPPER_BBOX + (0.95,)]))

        self.assertEqual(CellState.REDET, redetected.detections[0].state)
        self.assertEqual(1, redetected.detections[0].level)
        self.assertEqual(3, counter.total)

    def test_occlusion_pair_over_existing_stack_does_not_create_third_box(self) -> None:
        counter = self._interlocked_counter()
        self._stack_upper_cell(counter)
        pair = [
            self._UPPER_BBOX + (0.70,),
            self._PARTIAL_BBOX + (0.98,),
        ]

        repeated = counter.update(_still(pair))

        by_bbox = {d.bbox: d for d in repeated.detections}
        self.assertEqual(3, counter.total)
        self.assertEqual(CellState.REDET, by_bbox[self._UPPER_BBOX].state)
        self.assertEqual(1, by_bbox[self._UPPER_BBOX].level)
        self.assertEqual(CellState.VALIDATION, by_bbox[self._PARTIAL_BBOX].state)
        self.assertEqual(0, by_bbox[self._PARTIAL_BBOX].level)


class GateTests(PalletizingTestCase):
    """El gate vive en el cerebro, asi que se testea sin camara ni YOLO."""

    def test_arm_in_scene_blocks_counting(self) -> None:
        counter = GridCounter(self.roi, self.cfg)

        result = counter.update(FrameInput([(10, 40, 30, 60, 0.95)],
                                           arm_visible=True, motion_score=0.0))

        self.assertEqual(GateState.ARM_PAUSE, result.gate)
        self.assertEqual([], result.detections)
        self.assertEqual(0, counter.total)

    def test_motion_holds_counting_until_scene_settles(self) -> None:
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.gate = _gate(motion_pause_enabled=True, motion_stable_frames=2)
        counter = GridCounter(self.roi, cfg)
        boxes = [(10, 40, 30, 60, 0.95)]

        moving = counter.update(FrameInput(boxes, arm_visible=False, motion_score=99.0))
        settling = counter.update(_still(boxes))
        counting = counter.update(_still(boxes))

        self.assertEqual(GateState.MOTION_PAUSE, moving.gate)
        self.assertEqual(GateState.SETTLING, settling.gate)
        self.assertEqual(GateState.COUNTING, counting.gate)
        self.assertEqual(1, counter.total)

    def test_single_dropped_arm_frame_does_not_close_the_cycle(self) -> None:
        """El bug que motivo arm_debounce_frames: un unico frame en que el
        detector pierde el brazo se leia como viaje terminado y habilitaba
        contar otra caja dentro del mismo viaje."""
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.gate = _gate(arm_debounce_frames=3)
        counter = GridCounter(self.roi, cfg)

        counter.update(FrameInput([], arm_visible=True, motion_score=0.0))
        counter.update(FrameInput([], arm_visible=False, motion_score=0.0))  # parpadeo
        counter.update(FrameInput([], arm_visible=True, motion_score=0.0))

        self.assertFalse(counter.arm_cycle_seen)

    def test_sustained_absence_closes_the_cycle(self) -> None:
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.gate = _gate(arm_debounce_frames=3)
        counter = GridCounter(self.roi, cfg)

        counter.update(FrameInput([], arm_visible=True, motion_score=0.0))
        for _ in range(3):
            counter.update(FrameInput([], arm_visible=False, motion_score=0.0))

        self.assertTrue(counter.arm_cycle_seen)

    def test_first_stable_template_scene_after_arm_is_initial_baseline(self) -> None:
        """Arrancar a mitad de operacion no convierte el snapshot en colocadas."""
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.overlap_warn_ratio = 0.15
        counter = GridCounter(self.roi, cfg)
        counter.set_box_class("coin_roll_10")
        counter._close_arm_cycle()

        baseline = counter.update(_still([(10, 40, 30, 60, 0.95)]))

        self.assertEqual(CellState.NEW, baseline.detections[0].state)
        self.assertEqual(1, counter.initial)
        self.assertEqual(0, counter.placed)
        self.assertEqual(0, counter._placement_credits)

        counter._close_arm_cycle()
        counter.update(_still([
            (10, 40, 30, 60, 0.95),
            (70, 30, 90, 70, 0.94),
        ]))

        self.assertEqual(1, counter.placed)

    def test_missed_cycle_credit_recovers_two_box_backlog_later(self) -> None:
        """Cada viaje terminado concede una colocacion aunque la vision no
        confirme la caja en ese ciclo. Dos creditos permiten recuperar juntas
        la omitida y la caja nueva del ciclo siguiente."""
        counter = GridCounter(self.roi, self.cfg)

        counter._close_arm_cycle()  # caja fisica omitida por vision
        counter._close_arm_cycle()  # siguiente caja fisica
        recovered = counter.update(_still([
            (5, 30, 25, 70, 0.95),
            (70, 30, 90, 70, 0.94),
        ]))

        self.assertEqual([CellState.NEW, CellState.NEW], [
            detection.state for detection in recovered.detections
        ])
        self.assertEqual(2, counter.total)
        self.assertEqual(2, counter.placed)
        self.assertEqual(0, counter._placement_credits)

        without_credit = counter.update(_still([(38, 30, 58, 70, 0.93)]))
        self.assertEqual(CellState.REJECTED, without_credit.detections[0].state)
        self.assertEqual("ciclo-brazo", without_credit.detections[0].reason)


class EmptyPalletResetTests(PalletizingTestCase):
    """La paleta se vacia en la realidad: N frames COUNTING seguidos sin
    ninguna deteccion, habiendo cajas confirmadas, tienen que resetear TODO
    el conteo (GridCounter.reset_pallet(), disparado desde _count_boxes()).
    Es la unica transicion 1->0 de chi(g,z) de todo el paquete."""

    def _counter(self, empty_pallet_debounce_frames: int = 3, **gate_overrides) -> GridCounter:
        cfg = SimpleNamespace(**vars(self.cfg))
        cfg.gate = _gate(
            empty_pallet_debounce_frames=empty_pallet_debounce_frames, **gate_overrides,
        )
        return GridCounter(self.roi, cfg)

    def test_empty_frames_below_threshold_do_not_reset_a_confirmed_box(self) -> None:
        counter = self._counter(empty_pallet_debounce_frames=3)
        counter.update(_still([(10, 40, 30, 60, 0.95)]))
        self.assertEqual(1, counter.total)

        for _ in range(2):  # uno menos que el umbral
            counter.update(_still([]))

        self.assertEqual(1, counter.total)
        self.assertEqual(1, len(counter._occupied))

    def test_sustained_empty_frames_reset_the_confirmed_count(self) -> None:
        counter = self._counter(empty_pallet_debounce_frames=3)
        counter.update(_still([(10, 40, 30, 60, 0.95)]))
        self.assertEqual(1, counter.total)

        for _ in range(3):
            counter.update(_still([]))

        self.assertEqual(0, counter.total)
        self.assertEqual(0, counter.initial)
        self.assertEqual(0, counter.placed)
        self.assertEqual(set(), counter._occupied)

    def test_reset_frame_reports_count_changed_and_yields_an_empty_scene(self) -> None:
        counter = self._counter(empty_pallet_debounce_frames=2)
        counter.update(_still([(10, 40, 30, 60, 0.95)]))
        counter.update(_still([]))  # 1/2, todavia no dispara
        triggering = counter.update(_still([]))  # 2/2 -> reset

        self.assertTrue(triggering.count_changed)
        scene = counter.scene_state(0.6)
        self.assertEqual([], scene.boxes)
        self.assertEqual(0, scene.total)

    def test_reset_allows_a_fresh_pallet_to_be_counted_from_scratch(self) -> None:
        counter = self._counter(empty_pallet_debounce_frames=2)
        counter.update(_still([(10, 40, 30, 60, 0.95)]))
        for _ in range(2):
            counter.update(_still([]))
        self.assertEqual(0, counter.total)

        result = counter.update(_still([(10, 40, 30, 60, 0.95)]))

        self.assertEqual(CellState.NEW, result.detections[0].state)
        self.assertEqual(1, counter.total)
        self.assertEqual(1, counter.initial)

    def test_a_continuously_visible_box_never_triggers_a_reset(self) -> None:
        counter = self._counter(empty_pallet_debounce_frames=2)
        box = (10, 40, 30, 60, 0.95)
        for _ in range(10):
            counter.update(_still([box]))

        self.assertEqual(1, counter.total)
        self.assertEqual(0, counter._empty_frames)

    def test_reset_clears_arm_cycle_progress_for_the_next_pallet(self) -> None:
        counter = self._counter(empty_pallet_debounce_frames=2)
        counter.update(_still([(10, 40, 30, 60, 0.95)]))
        counter._close_arm_cycle()
        self.assertTrue(counter.arm_cycle_seen)
        self.assertEqual(1, counter._placement_credits)

        for _ in range(2):
            counter.update(_still([]))

        self.assertFalse(counter.arm_cycle_seen)
        self.assertEqual(0, counter._placement_credits)

    def test_reset_does_not_touch_camera_calibration(self) -> None:
        counter = self._counter(empty_pallet_debounce_frames=2)
        homography_before = counter._homography
        ladder_before = counter._ladder
        counter.update(_still([(10, 40, 30, 60, 0.95)]))

        for _ in range(2):
            counter.update(_still([]))

        self.assertIs(counter._homography, homography_before)
        self.assertIs(counter._ladder, ladder_before)

    def test_reset_does_not_fire_while_gate_is_not_counting(self) -> None:
        """Con el brazo en escena _count_boxes() ni se ejecuta -- el contador
        de vacio (y por lo tanto el reset) es exclusivo del gate COUNTING."""
        counter = self._counter(empty_pallet_debounce_frames=2, arm_debounce_frames=1)
        counter.update(_still([(10, 40, 30, 60, 0.95)]))
        self.assertEqual(1, counter.total)

        for _ in range(10):
            result = counter.update(FrameInput([], arm_visible=True, motion_score=0.0))
            self.assertEqual(GateState.ARM_PAUSE, result.gate)

        self.assertEqual(1, counter.total)
        self.assertEqual(0, counter._empty_frames)


class SceneStateTests(PalletizingTestCase):
    """La unificacion de footprint por nivel vivia en isometric.py; ahora es
    del cerebro y se testea aqui (ver test_isometric_uniform_footprint.py en
    el historial de git)."""

    def test_level_consensus_is_applied_preserving_each_box_orientation(self) -> None:
        counter = GridCounter(self.roi, self.cfg)
        # La del medio esta girada 90 grados respecto a las otras dos.
        counter.update(_still([
            (5, 40, 35, 60, 0.95),
            (45, 25, 65, 75, 0.94),
            (70, 40, 100, 60, 0.93),
        ]))

        scene = counter.scene_state(0.6)

        self.assertEqual(3, len(scene.boxes))
        long_side = max(scene.boxes[0].side_a, scene.boxes[0].side_b)
        short_side = min(scene.boxes[0].side_a, scene.boxes[0].side_b)
        for box in scene.boxes:
            self.assertAlmostEqual(long_side, max(box.side_a, box.side_b), places=6)
            self.assertAlmostEqual(short_side, min(box.side_a, box.side_b), places=6)
        # La caja girada conserva su orientacion: su lado a es el corto.
        self.assertLess(scene.boxes[1].side_a, scene.boxes[1].side_b)
        self.assertGreater(scene.boxes[0].side_a, scene.boxes[0].side_b)

    def test_level_tops_stack_and_report_total_height(self) -> None:
        counter = GridCounter(self.roi, self.two_level_cfg())
        counter._bootstrap_reconciled = True
        counter.update(_still([(10, 40, 30, 60, 0.95)]))

        scene = counter.scene_state(0.6)

        self.assertEqual(scene.levels + 1, len(scene.level_tops))
        self.assertEqual(0.0, scene.level_tops[0])
        self.assertAlmostEqual(scene.total_height, scene.level_tops[-1], places=6)
        self.assertEqual(0.0, scene.boxes[0].z0)

    def test_empty_pallet_yields_empty_scene(self) -> None:
        counter = GridCounter(self.roi, self.cfg)

        scene = counter.scene_state(0.6)

        self.assertEqual([], scene.boxes)
        self.assertEqual([], scene.overlaps)
        self.assertEqual(0, scene.total)

    def test_pallet_state_json_round_trip_restores_3d_identity(self) -> None:
        counter = GridCounter(self.roi, self.cfg)
        counter.set_box_class("test_box")
        counter.update(_still([
            (10, 30, 40, 70, 0.95),
            (60, 30, 90, 70, 0.94),
        ]))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera_3.json"
            counter.save_state(path, height_ratio=0.6)
            restored = GridCounter(self.roi, self.cfg)
            restored.set_box_class("test_box")
            restored.load_state(path)

            self.assertEqual(counter.scene_state(0.6), restored.scene_state(0.6))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(1, payload["schema_version"])
            self.assertEqual("test_box", payload["active_box_class"])
            self.assertEqual(2, len(payload["levels"][0]["boxes"]))
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_pallet_state_round_trip_restores_pending_placement_credits(self) -> None:
        counter = GridCounter(self.roi, self.cfg)
        counter.set_box_class("test_box")
        counter._close_arm_cycle()
        counter._close_arm_cycle()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera_3.json"
            counter.save_state(path, height_ratio=0.6)
            restored = GridCounter(self.roi, self.cfg)
            restored.set_box_class("test_box")
            restored.load_state(path)

            self.assertTrue(restored.arm_cycle_seen)
            self.assertEqual(2, restored._placement_credits)

    def test_invalid_state_does_not_mutate_counter(self) -> None:
        counter = GridCounter(self.roi, self.cfg)
        counter.set_box_class("test_box")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")

            with self.assertRaises(ValueError):
                counter.load_state(path)

        self.assertEqual(0, counter.total)
        self.assertEqual(set(), counter._occupied)


if __name__ == "__main__":
    unittest.main()
