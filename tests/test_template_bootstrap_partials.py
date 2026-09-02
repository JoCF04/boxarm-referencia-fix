from dataclasses import replace
from pathlib import Path
import logging
from statistics import median

import numpy as np

from boxarm.config import load_palletizing_config
from boxarm.vision.palletizing import GridCounter
from boxarm.vision.palletizing.templates.template_matcher import (
    TemplateFit,
    TemplateObservation,
)
from boxarm.vision.palletizing.templates.template_runtime import get_layout_template


def _counter() -> GridCounter:
    path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    cfg = replace(load_palletizing_config(path))
    roi = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.int32)
    counter = GridCounter(roi, cfg)
    counter.set_box_class("coin_roll_100")
    counter._level_footprint[0] = (0.32, 0.15)
    return counter


def _observation(slot) -> TemplateObservation:
    return TemplateObservation(slot.u, slot.v, slot.width, slot.height)


def _split_vertically(slot) -> tuple[TemplateObservation, TemplateObservation]:
    half = slot.width / 2
    offset = slot.width / 4
    return (
        TemplateObservation(slot.u - offset, slot.v, half, slot.height),
        TemplateObservation(slot.u + offset, slot.v, half, slot.height),
    )


def _detection(observation: TemplateObservation):
    x1 = int(round((observation.u - observation.width / 2) * 1000))
    y1 = int(round((observation.v - observation.height / 2) * 1000))
    x2 = int(round((observation.u + observation.width / 2) * 1000))
    y2 = int(round((observation.v + observation.height / 2) * 1000))
    return (x1, y1, x2, y2, 0.95, "coin_roll_100")


def test_two_complementary_fragments_infer_free_slot_and_apply_fifteen() -> None:
    counter = _counter()
    template = get_layout_template("coin_roll_100", 0, 0)
    assert template is not None
    observations = tuple(_observation(slot) for slot in template.slots[:-1])
    partials = _split_vertically(template.slots[-1])

    fit = counter._select_template_bootstrap_fit(observations, partials, 0)

    assert fit is not None
    assert {slot.cell for level, slot in fit.inferred_slots if level == 0} == {14}
    counter._apply_template_bootstrap_fit(fit, observations, 0)
    assert counter.total == 15
    assert sum(level == 0 for _cell, level in counter._occupied) == 15


def test_inferred_slot_participates_in_temporal_signature() -> None:
    counter = _counter()
    template = get_layout_template("coin_roll_100", 0, 0)
    assert template is not None
    observations = tuple(_observation(slot) for slot in template.slots[:-2])

    first = TemplateFit(
        assignments=(), mean_error=0.0, phase=0,
        inferred_slots=((0, template.slots[-1]),),
    )
    second = TemplateFit(
        assignments=(), mean_error=0.0, phase=0,
        inferred_slots=((0, template.slots[-2]),),
    )

    assert counter._template_bootstrap_signature(first, observations, 0) != (
        counter._template_bootstrap_signature(second, observations, 0)
    )


def test_fourteen_complete_plus_two_fragments_apply_only_after_min_stable() -> None:
    counter = _counter()
    template = get_layout_template("coin_roll_100", 0, 0)
    assert template is not None
    observations = tuple(_observation(slot) for slot in template.slots[:-1])
    partials = _split_vertically(template.slots[-1])
    detections = [_detection(item) for item in (*observations, *partials)]

    for _ in range(counter._cfg.confirmation.min_stable - 1):
        counter._count_boxes(detections)
        assert not counter._bootstrap_reconciled

    counter._count_boxes(detections)

    assert counter._bootstrap_reconciled
    assert counter.total == 15
    assert counter.initial == 15


def test_partial_inside_assigned_complete_is_duplicate_not_second_level() -> None:
    counter = _counter()
    template = get_layout_template("coin_roll_100", 0, 0)
    assert template is not None
    observations = tuple(_observation(slot) for slot in template.slots)
    duplicate = _split_vertically(template.slots[3])[0]

    fit = counter._select_template_bootstrap_fit(observations, (duplicate,), 0)

    assert fit is not None
    assert fit.inferred_slots == ()
    assert {assignment.level for assignment in fit.assignments} == {0}


def test_random_partial_does_not_validate_one_level() -> None:
    counter = _counter()
    template = get_layout_template("coin_roll_100", 0, 0)
    assert template is not None
    observations = tuple(_observation(slot) for slot in template.slots[:-1])
    random_partial = TemplateObservation(0.98, 0.50, 0.02, 0.02)

    assert counter._select_template_bootstrap_fit(
        observations, (random_partial,), 0,
    ) is None


def test_explained_two_levels_still_win_when_one_level_cannot_fit() -> None:
    counter = _counter()
    lower = get_layout_template("coin_roll_100", 0, 0)
    upper = get_layout_template("coin_roll_100", 1, 0)
    assert lower is not None and upper is not None
    observations = tuple(
        [_observation(slot) for slot in lower.slots]
        + [_observation(upper.slots[0])]
    )
    # El slot 0 inferior queda realmente ocluido por el slot 0 superior en
    # esta pareja A/B; por eso el fragmento tiene explicacion fisica en N0.
    covered_lower = lower.slots[0]
    partial = _split_vertically(covered_lower)[0]

    fit = counter._select_template_bootstrap_fit(observations, (partial,), 0)

    assert fit is not None
    assert any(assignment.level == 1 for assignment in fit.assignments)


def test_template_fit_constructor_remains_backward_compatible() -> None:
    fit = TemplateFit(assignments=(), mean_error=0.0, phase=0)
    assert fit.inferred_slots == ()


def test_no_fit_log_is_emitted_only_when_problem_signature_changes(
    monkeypatch, caplog,
) -> None:
    counter = _counter()
    monkeypatch.setattr(
        counter, "_select_template_bootstrap_fit", lambda *_args: None,
    )
    detections = []
    for index in range(8):
        x = 30 + (index % 4) * 230
        y = 30 + (index // 4) * 210
        detections.append(((x, y, x + 245, y + 128), 0.95, "coin_roll_100"))

    with caplog.at_level(logging.DEBUG, logger="boxarm.vision.palletizing.bootstrap"):
        counter._reconcile_initial_layers(detections)
        counter._reconcile_initial_layers(detections)
        moved = list(detections)
        (x1, y1, x2, y2), confidence, cls_name = moved[0]
        moved[0] = ((x1 + 20, y1, x2 + 20, y2), confidence, cls_name)
        counter._reconcile_initial_layers(moved)

    messages = [
        record.message for record in caplog.records
        if "ninguna configuracion valida" in record.message
    ]
    assert len(messages) == 2
    assert "candidatos una=0 dos=0 explicados_una=0 explicados_dos=0" in messages[0]


def test_second_100_frame_resolves_25_lower_and_9_upper_without_waiting() -> None:
    """Regresion real de Camara 2: 28 visibles representan 34 fisicas."""
    path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    cfg = replace(
        load_palletizing_config(path),
        max_bootstrap_combinations=200,
    )
    roi = np.array(
        [[487, 64], [872, 58], [879, 515], [497, 522]],
        dtype=np.int32,
    )
    counter = GridCounter(roi, cfg)
    counter.set_box_class("coin_roll_10")

    # Detecciones unicas de YOLO en p010soles.avi @ 100.0 s. Se quitaron
    # tres duplicados de la salida cruda (31 -> 28 posiciones visibles).
    boxes = [
        (776, 66, 873, 127), (706, 72, 777, 131),
        (604, 74, 699, 134), (507, 75, 601, 136),
        (777, 131, 875, 192), (706, 135, 778, 193),
        (603, 135, 698, 194), (505, 137, 600, 198),
        (707, 196, 780, 255), (780, 196, 877, 256),
        (607, 199, 703, 260), (507, 202, 602, 262),
        (782, 260, 879, 320), (710, 262, 781, 320),
        (607, 264, 703, 323), (508, 265, 603, 325),
        (711, 324, 782, 383), (782, 326, 879, 386),
        (609, 327, 705, 387), (509, 328, 605, 388),
        (713, 393, 782, 452), (512, 395, 606, 454),
        (610, 395, 706, 454), (783, 393, 879, 456),
        (512, 457, 572, 515), (780, 457, 878, 518),
        (676, 460, 774, 521), (571, 461, 668, 523),
    ]
    full_measures = []
    for x1, y1, x2, y2 in boxes:
        if x2 - x1 >= 90:
            full_measures.append(counter._measure_footprint(
                (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1,
            ))
    counter._level_footprint[0] = (
        median(max(item) for item in full_measures),
        median(min(item) for item in full_measures),
    )
    detections = [(bbox, 0.95, "coin_roll_10") for bbox in boxes]

    counter._reconcile_initial_layers(detections)

    assert counter._bootstrap_reconciled
    assert sum(level == 0 for _cell, level in counter._occupied) == 25
    assert sum(level == 1 for _cell, level in counter._occupied) == 9
    assert counter.total == 34
