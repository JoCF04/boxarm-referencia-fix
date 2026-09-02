from pathlib import Path

import numpy as np

from boxarm.config import load_palletizing_config, load_pipeline_config
from boxarm.vision.inference import _bbox_fully_inside_roi, _scale_roi
from boxarm.vision.palletizing import GridCounter
from boxarm.vision.palletizing.bootstrap import (
    _BootstrapMixin,
    _template_min_evidence,
)
from boxarm.vision.palletizing.templates.template_matcher import (
    TemplateObservation,
    fit_layout_hypothesis,
)
from boxarm.vision.palletizing.templates.template_runtime import get_layout_template


def _transformed_observation(slot, registration):
    sx, sy, tx, ty = registration
    return TemplateObservation(
        u=slot.u * sx + tx,
        v=slot.v * sy + ty,
        width=slot.width * sx,
        height=slot.height * sy,
    )


def test_template_fit_registers_partial_layout_inside_a_larger_roi() -> None:
    template = get_layout_template("coin_roll_10", level=0, phase=0)
    assert template is not None
    expected_registration = (0.70, 0.76, 0.15, 0.07)
    observations = tuple(
        _transformed_observation(template.slots[index], expected_registration)
        for index in (0, 2, 6, 10, 14, 18)
    )

    fit = fit_layout_hypothesis(
        "coin_roll_10",
        base_level=0,
        observations=observations,
        include_upper=False,
        max_center_distance=0.12,
        min_side_ratio=0.70,
        phase=0,
    )

    assert fit is not None
    assert fit.registration is not None
    assert {assignment.slot.cell for assignment in fit.assignments} == {
        0, 2, 6, 10, 14, 18,
    }


def test_template_fit_accepts_registration_in_raw_image_pixels() -> None:
    template = get_layout_template("coin_roll_10", level=0, phase=0)
    assert template is not None
    expected_registration = (392.0, 469.0, 457.0, 43.0)
    observations = tuple(
        _transformed_observation(template.slots[index], expected_registration)
        for index in (0, 2, 6, 10, 14, 18, 22)
    )

    fit = fit_layout_hypothesis(
        "coin_roll_10",
        base_level=0,
        observations=observations,
        include_upper=False,
        max_center_distance=0.12,
        min_side_ratio=0.70,
        phase=0,
    )

    assert fit is not None
    assert fit.registration is not None
    assert {assignment.slot.cell for assignment in fit.assignments} == {
        0, 2, 6, 10, 14, 18, 22,
    }


def test_template_fit_is_not_combinatorial_with_legacy_state_budget() -> None:
    """El matching polinomial no depende del antiguo presupuesto recursivo."""
    template = get_layout_template("coin_roll_10", level=0, phase=0)
    assert template is not None
    observations = tuple(
        _transformed_observation(template.slots[index], (392.0, 469.0, 457.0, 43.0))
        for index in (0, 2, 6, 10, 14, 18, 22)
    )

    fit = fit_layout_hypothesis(
        "coin_roll_10",
        base_level=0,
        observations=observations,
        include_upper=False,
        max_center_distance=0.12,
        min_side_ratio=0.70,
        phase=0,
        max_assignment_states=1,
    )

    assert fit is not None


def test_bag_template_fit_tolerates_perspective_size_change() -> None:
    """Una caja completa no deja de serlo por la perspectiva de la camara."""
    observations = tuple(
        TemplateObservation(u, v, width, height)
        for u, v, width, height in (
            # Detecciones completas reales del frame 0 de Camara 3.
            (579.0, 444.5, 176.0, 273.0),
            (760.5, 439.5, 175.0, 273.0),
            (573.0, 107.0, 186.0, 214.0),
            (409.0, 444.5, 172.0, 267.0),
            (393.0, 107.5, 180.0, 215.0),
            (718.0, 214.5, 274.0, 177.0),
        )
    )

    fits = [
        fit_layout_hypothesis(
            "bag_10",
            base_level=0,
            observations=observations,
            include_upper=True,
            max_center_distance=0.18,
            min_side_ratio=0.80,
            phase=phase,
        )
        for phase in (0, 1)
    ]

    assert any(fit is not None for fit in fits)


def test_camera_3_bag_frame_resolves_two_levels_after_stability() -> None:
    root = Path(__file__).resolve().parents[1]
    pipeline = load_pipeline_config(root / "configs" / "pipeline.yaml")
    camera = next(item for item in pipeline.cameras if item.id == 3)
    roi = _scale_roi(camera.roi, 1280, 720)
    counter = GridCounter(
        roi,
        load_palletizing_config(root / "configs" / "palletizing.yaml"),
    )
    counter.set_box_class("bag_10")
    boxes = (
        (493, 308, 665, 581),
        (675, 306, 848, 575),
        (324, 313, 493, 578),
        (304, 0, 481, 216),
        (482, 0, 665, 216),
        (666, 0, 845, 126),
        (583, 126, 853, 300),
        (316, 213, 578, 304),
    )
    footprints = []
    for x1, y1, x2, y2 in boxes:
        footprints.append(counter._measure_footprint(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1,
        ))
    counter._level_footprint[0] = (
        float(np.median([item[0] for item in footprints])),
        float(np.median([item[1] for item in footprints])),
    )
    parsed = [(bbox, 0.99, "bag_10") for bbox in boxes]

    for _ in range(counter._cfg.confirmation.min_stable):
        counter._reconcile_initial_layers(parsed)

    assert counter._bootstrap_reconciled
    assert counter._template_phase == 1
    assert sum(level == 0 for _cell, level in counter._occupied) == 6
    assert sum(level == 1 for _cell, level in counter._occupied) == 2
    assert counter.total == 8


def test_bootstrap_threshold_is_derived_from_class_capacity() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    load_palletizing_config(config_path)

    assert _template_min_evidence(25) == 13
    assert _template_min_evidence(18) == 10
    assert _template_min_evidence(15) == 8
    assert _template_min_evidence(7) == 4


def test_template_bootstrap_removes_equivalent_duplicate_boxes() -> None:
    observations = [
        TemplateObservation(0.20, 0.30, 0.18, 0.10),
        TemplateObservation(0.202, 0.301, 0.179, 0.101),
        TemplateObservation(0.50, 0.30, 0.18, 0.10),
    ]

    unique = _BootstrapMixin._deduplicate_template_observations(observations)

    assert len(unique) == 2


def test_half_capacity_stays_visible_without_analysis_until_threshold() -> None:
    cfg = load_palletizing_config(
        Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml",
    )
    counter = GridCounter(
        np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.int32),
        cfg,
    )
    counter.set_box_class("coin_roll_10")
    counter._level_footprint[0] = (0.245, 0.128)
    template = get_layout_template("coin_roll_10", level=0, phase=0)
    assert template is not None

    def parsed(count):
        result = []
        for slot in template.slots[:count]:
            result.append(((
                int((slot.u - slot.width / 2) * 1000),
                int((slot.v - slot.height / 2) * 1000),
                int((slot.u + slot.width / 2) * 1000),
                int((slot.v + slot.height / 2) * 1000),
            ), 0.99, "coin_roll_10"))
        return result

    counter._reconcile_initial_layers(parsed(12))
    assert not counter._initial_scene_deferred

    counter._reconcile_initial_layers(parsed(13))
    assert counter._initial_scene_deferred


def test_roi_requires_the_complete_bbox_not_only_its_center() -> None:
    roi = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.int32)

    assert _bbox_fully_inside_roi((20, 20, 80, 80), roi)
    assert not _bbox_fully_inside_roi((5, 20, 80, 80), roi)
