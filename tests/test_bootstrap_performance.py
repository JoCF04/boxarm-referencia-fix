from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from boxarm.config import load_palletizing_config
from boxarm.vision.palletizing import GridCounter
from boxarm.vision.palletizing import bootstrap
from boxarm.vision.palletizing.formulas import _rect_union_coverage


def _counter() -> GridCounter:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    base = load_palletizing_config(config_path)
    boxes_per_level = dict(base.boxes_per_level)
    boxes_per_level["test_box"] = 15
    cfg = replace(base, boxes_per_level=boxes_per_level)
    roi = np.array(
        [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
        dtype=np.int32,
    )
    counter = GridCounter(roi, cfg)
    counter.set_box_class("test_box")
    counter._level_footprint[0] = (0.245, 0.128)
    return counter


def _template_counter(
    yaml_capacity: int | None,
    *,
    configured_min_boxes: int = 1,
) -> GridCounter:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    base = load_palletizing_config(config_path)
    boxes_per_level = dict(base.boxes_per_level)
    if yaml_capacity is None:
        boxes_per_level.pop("coin_roll_100", None)
    else:
        boxes_per_level["coin_roll_100"] = yaml_capacity
    cfg = replace(
        base,
        boxes_per_level=boxes_per_level,
        template_bootstrap_min_boxes=configured_min_boxes,
    )
    roi = np.array(
        [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
        dtype=np.int32,
    )
    counter = GridCounter(roi, cfg)
    counter.set_box_class("coin_roll_100")
    counter._level_footprint[0] = (0.245, 0.128)
    return counter


def _template_detections(
    complete: int,
    *,
    partial: int = 0,
) -> list[tuple[tuple[int, int, int, int], float, str]]:
    detections = []
    for index in range(complete):
        x = 30 + (index % 4) * 230
        y = 30 + (index // 4) * 210
        detections.append(((x, y, x + 245, y + 128), 0.95, "coin_roll_100"))
    for index in range(partial):
        x = 60 + index * 80
        detections.append(((x, 700, x + 40, 740), 0.90, "coin_roll_100"))
    return detections


@pytest.mark.parametrize("yaml_capacity", [None, 99])
def test_template_capacity_is_authoritative_when_yaml_is_missing_or_wrong(
    monkeypatch,
    yaml_capacity: int | None,
) -> None:
    counter = _template_counter(yaml_capacity)
    calls: list[tuple[int, int, int]] = []

    def capture_fit(observations, partials, base_level):
        calls.append((len(observations), len(partials), base_level))
        return None

    monkeypatch.setattr(counter, "_select_template_bootstrap_fit", capture_fit)

    counter._reconcile_initial_layers(_template_detections(complete=8))

    assert calls == [(8, 0, 0)]


def test_template_bootstrap_requires_strictly_more_than_half_the_slots(
    monkeypatch,
) -> None:
    counter = _template_counter(15, configured_min_boxes=1)
    calls: list[int] = []

    def capture_fit(observations, partials, _base_level):
        calls.append(len(observations) + len(partials))
        return None

    monkeypatch.setattr(counter, "_select_template_bootstrap_fit", capture_fit)

    counter._reconcile_initial_layers(_template_detections(complete=7))
    assert calls == []

    counter._reconcile_initial_layers(_template_detections(complete=8))
    assert calls == [8]


def test_template_class_never_dispatches_to_generic_bootstrap(
    monkeypatch,
) -> None:
    counter = _template_counter(99)
    template_calls = 0

    def no_fit(_observations, _partials, _base_level):
        nonlocal template_calls
        template_calls += 1
        return None

    def fail_generic(*_args, **_kwargs):
        pytest.fail("una clase con plantilla no debe entrar al bootstrap generico")

    monkeypatch.setattr(counter, "_select_template_bootstrap_fit", no_fit)
    monkeypatch.setattr(counter, "_bootstrap_partial_hypotheses", fail_generic)

    counter._reconcile_initial_layers(
        _template_detections(complete=7, partial=1),
    )

    assert template_calls == 1


def test_rect_union_coverage_index_preserves_raster_formula_exactly() -> None:
    """El indice rapido debe conservar los mismos pixeles que la formula F7."""
    rects = [
        (0.333, 0.778, 0.245, 0.128),
        (0.257, 0.598, 0.128, 0.245),
        (0.660, 0.469, 0.128, 0.245),
    ]
    targets = [
        (0.333, 0.778, 0.245, 0.128),
        (0.257, 0.6555, 0.128, 0.245),
        (0.3795, 0.778, 0.245, 0.128),
        (0.05, 0.05, 0.20, 0.20),
        (1.10, 1.10, 0.10, 0.10),
    ]

    index = bootstrap._RectUnionCoverageIndex(rects, occupancy_grid=200)

    assert [index.coverage(target) for target in targets] == [
        _rect_union_coverage(target, rects, 200) for target in targets
    ]


def test_hidden_candidate_search_does_not_rasterize_union_per_candidate(
    monkeypatch,
) -> None:
    """Regresion estructural: un mismo conjunto superior se rasteriza una vez.

    El caso real de Camara 1 generaba 286 rasterizaciones completas solo para
    una topologia promovida. Con cientos de topologias, ese trabajo repetido
    llevaba el bootstrap a varios segundos por frame.
    """
    counter = _counter()
    upper = [
        (0.333, 0.778, 0.245, 0.128),
        (0.257, 0.598, 0.128, 0.245),
    ]
    repeated_full_rasterizations = 0
    original = bootstrap._rect_union_coverage

    def counted(*args):
        nonlocal repeated_full_rasterizations
        repeated_full_rasterizations += 1
        return original(*args)

    monkeypatch.setattr(bootstrap, "_rect_union_coverage", counted)

    candidates = counter._bootstrap_hidden_candidates(
        [], upper, (0.245, 0.128), level=0,
    )

    assert len(candidates) == 20
    assert repeated_full_rasterizations <= 1
