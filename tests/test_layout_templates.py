import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from boxarm.config import load_palletizing_config
from boxarm.vision.palletizing import GridCounter
from boxarm.vision.palletizing.layout_templates import (
    BoxOrientation,
    TemplateObservation,
    fit_layout_hypothesis,
    get_layout_template,
    match_layout_slot,
    render_layout_templates,
)
from boxarm.vision.palletizing.templates.coin_roll_100 import PATTERN_A, PATTERN_B
from boxarm.vision.palletizing.templates.dsl import HORIZONTAL, VERTICAL


def test_coin_roll_100_runtime_is_compiled_exactly_from_class_data() -> None:
    """La capa runtime no debe corregir, reordenar ni redondear el DSL."""
    for phase, raw_pattern in enumerate((PATTERN_A, PATTERN_B)):
        template = get_layout_template("coin_roll_100", level=0, phase=phase)

        assert template is not None
        assert tuple(
            (
                slot.u,
                slot.v,
                slot.width,
                slot.height,
                HORIZONTAL
                if slot.orientation is BoxOrientation.HORIZONTAL
                else VERTICAL,
            )
            for slot in template.slots
        ) == raw_pattern


def test_pattern_a_matches_the_single_floor_physical_topology() -> None:
    """El frame 02:20 mezcla dos pisos; no define por si solo Pattern A."""
    template = get_layout_template("coin_roll_100", level=0, phase=0)
    assert template is not None
    slots = {slot.cell: slot for slot in template.slots}

    horizontal = {0, 1, 5, 10, 13, 14}
    assert {
        cell
        for cell, slot in slots.items()
        if slot.orientation is BoxOrientation.HORIZONTAL
    } == horizontal

    # Croquis fisico: 03 y 11 forman la segunda columna vertical;
    # 06 y 08 son las dos verticales centrales contiguas.
    assert slots[2].u < slots[3].u < slots[6].u < slots[8].u < slots[4].u
    assert slots[7].u < slots[11].u < slots[6].u
    assert abs(slots[6].v - slots[8].v) < 0.03

    # 05 va bajo la fila superior; 10 va encima de las dos cajas inferiores.
    assert slots[0].v < slots[5].v < slots[10].v < slots[13].v
    assert slots[1].v < slots[5].v
    assert slots[14].v > slots[10].v


def test_public_facade_delegates_matching_and_rendering_to_separate_modules() -> None:
    assert fit_layout_hypothesis.__module__.endswith("template_matcher")
    assert match_layout_slot.__module__.endswith("template_matcher")
    assert get_layout_template.__module__.endswith("template_runtime")
    assert render_layout_templates.__module__.endswith("template_render")


def test_coin_roll_100_has_two_normalized_fifteen_slot_templates() -> None:
    even = get_layout_template("coin_roll_100", level=0)
    odd = get_layout_template("coin_roll_100", level=1)

    assert even is not None
    assert odd is not None
    assert len(even.slots) == 15
    assert len(odd.slots) == 15
    assert even != odd

    for template in (even, odd):
        assert [slot.cell for slot in template.slots] == list(range(15))
        for slot in template.slots:
            assert 0.0 <= slot.u <= 1.0
            assert 0.0 <= slot.v <= 1.0
            assert 0.0 < slot.width <= 1.0
            assert 0.0 < slot.height <= 1.0
            expected = (
                BoxOrientation.HORIZONTAL
                if slot.width >= slot.height
                else BoxOrientation.VERTICAL
            )
            assert slot.orientation is expected


def test_coin_roll_100_templates_repeat_by_level_parity() -> None:
    assert get_layout_template("coin_roll_100", 0) is get_layout_template("coin_roll_100", 2)
    assert get_layout_template("coin_roll_100", 1) is get_layout_template("coin_roll_100", 3)
    assert get_layout_template("coin_roll_100", 0, phase=1) is get_layout_template(
        "coin_roll_100", 1, phase=0,
    )
    assert get_layout_template("coin_roll_100", 1, phase=1) is get_layout_template(
        "coin_roll_100", 0, phase=0,
    )
    assert get_layout_template("coin_roll_10", 0) is None


def test_match_layout_slot_predicts_the_free_physical_hole() -> None:
    template = get_layout_template("coin_roll_100", level=0)
    assert template is not None
    target = template.slots[6]

    matched = match_layout_slot(
        template,
        center=(target.u + 0.008, target.v - 0.006),
        footprint=(target.width * 0.98, target.height * 1.02),
        occupied={0, 1, 2, 3, 4, 5},
        max_center_distance=0.12,
    )

    assert matched == target


def test_match_layout_slot_rejects_occupied_or_wrong_orientation() -> None:
    template = get_layout_template("coin_roll_100", level=0)
    assert template is not None
    target = template.slots[6]

    assert match_layout_slot(
        template,
        center=(target.u, target.v),
        footprint=(target.width, target.height),
        occupied={target.cell},
        # Aunque otro hueco libre entre en una tolerancia amplia, no se debe
        # correr la caja vecina: primero se identifica el hueco fisico y
        # despues se comprueba su ocupacion.
        max_center_distance=0.30,
    ) is None
    assert match_layout_slot(
        template,
        center=(target.u, target.v),
        footprint=(target.height, target.width),
        occupied=set(),
        max_center_distance=0.01,
    ) is None


def test_counter_assigns_coin_roll_100_to_template_cell_not_arrival_order() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    cfg = load_palletizing_config(config_path)
    counter = GridCounter(
        np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.int32),
        cfg,
    )
    counter.set_box_class("coin_roll_100")
    template = get_layout_template("coin_roll_100", level=0)
    assert template is not None
    target = template.slots[11]

    cell, reason = counter._assign_cell(
        target.u * 1000,
        target.v * 1000,
        level=0,
        footprint=(target.width, target.height),
    )

    assert reason == ""
    assert cell == target.cell


def _observation(slot, *, du: float = 0.0, dv: float = 0.0) -> TemplateObservation:
    return TemplateObservation(
        u=slot.u + du,
        v=slot.v + dv,
        width=slot.width,
        height=slot.height,
    )


def test_global_fit_assigns_shuffled_centroids_one_to_one() -> None:
    even = get_layout_template("coin_roll_100", 0)
    assert even is not None
    source = [even.slots[index] for index in (11, 2, 8, 0, 14)]

    fit = fit_layout_hypothesis(
        "coin_roll_100",
        base_level=0,
        observations=tuple(_observation(slot) for slot in source),
        include_upper=False,
        max_center_distance=0.12,
        min_side_ratio=0.70,
    )

    assert fit is not None
    assert {assignment.slot.cell for assignment in fit.assignments} == {
        slot.cell for slot in source
    }
    assert {assignment.level for assignment in fit.assignments} == {0}
    assert fit.mean_error == 0.0


def test_global_fit_uses_both_templates_when_visible_boxes_come_from_two_levels() -> None:
    even = get_layout_template("coin_roll_100", 0)
    odd = get_layout_template("coin_roll_100", 1)
    assert even is not None and odd is not None
    observations = (
        _observation(even.slots[2]),
        _observation(even.slots[12]),
        _observation(odd.slots[1]),
        _observation(odd.slots[6]),
    )

    one_level = fit_layout_hypothesis(
        "coin_roll_100", 0, observations, False, 0.12, 0.70,
    )
    two_levels = fit_layout_hypothesis(
        "coin_roll_100", 0, observations, True, 0.12, 0.70,
    )

    assert two_levels is not None
    assert {assignment.level for assignment in two_levels.assignments} == {0, 1}
    assert two_levels.mean_error == 0.0
    assert one_level is None or one_level.mean_error > two_levels.mean_error


def test_global_fit_can_start_with_pattern_b_as_level_zero() -> None:
    pattern_b = get_layout_template("coin_roll_100", 0, phase=1)
    assert pattern_b is not None
    observations = tuple(_observation(pattern_b.slots[index]) for index in (0, 4, 7, 11, 14))

    wrong_phase = fit_layout_hypothesis(
        "coin_roll_100", 0, observations, False, 0.12, 0.70, phase=0,
    )
    correct_phase = fit_layout_hypothesis(
        "coin_roll_100", 0, observations, False, 0.12, 0.70, phase=1,
    )

    assert correct_phase is not None
    assert correct_phase.phase == 1
    assert correct_phase.mean_error == 0.0
    assert wrong_phase is None or wrong_phase.mean_error > correct_phase.mean_error


def test_global_fit_rejects_centroid_that_fits_no_physical_hole() -> None:
    fit = fit_layout_hypothesis(
        "coin_roll_100",
        base_level=0,
        observations=(TemplateObservation(0.50, 0.50, 0.05, 0.40),),
        include_upper=True,
        max_center_distance=0.06,
        min_side_ratio=0.70,
    )

    assert fit is None


def test_bootstrap_template_fit_completes_hidden_lower_level() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    cfg = load_palletizing_config(config_path)
    counter = GridCounter(
        np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.int32),
        cfg,
    )
    counter.set_box_class("coin_roll_100")
    odd = get_layout_template("coin_roll_100", 1)
    assert odd is not None
    observations = tuple(_observation(odd.slots[index]) for index in (1, 6))
    fit = fit_layout_hypothesis(
        "coin_roll_100", 0, observations, True, 0.12, 0.70,
    )
    assert fit is not None

    counter._apply_template_bootstrap_fit(fit, observations, base_level=0)

    assert counter._bootstrap_reconciled
    assert counter.total == 17
    assert sum(level == 0 for _cell, level in counter._occupied) == 15
    assert sum(level == 1 for _cell, level in counter._occupied) == 2
    assert counter._level_is_full(0)
    scene = counter.scene_state(0.60)
    assert scene.total == 17
    assert len(scene.boxes) == 17
    assert sum(box.level == 0 for box in scene.boxes) == 15
    assert sum(box.level == 1 for box in scene.boxes) == 2


def test_reconcile_initial_layers_can_select_pattern_b_as_first_floor() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    cfg = load_palletizing_config(config_path)
    counter = GridCounter(
        np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.int32),
        cfg,
    )
    counter.set_box_class("coin_roll_100")
    counter._level_footprint[0] = (0.32, 0.15)
    odd = get_layout_template("coin_roll_100", 1)
    assert odd is not None
    selected = [odd.slots[index] for index in (1, 4, 6, 9, 13)]
    parsed = []
    for slot in selected:
        x1 = int((slot.u - slot.width / 2) * 1000)
        y1 = int((slot.v - slot.height / 2) * 1000)
        x2 = int((slot.u + slot.width / 2) * 1000)
        y2 = int((slot.v + slot.height / 2) * 1000)
        parsed.append(((x1, y1, x2, y2), 0.98, "coin_roll_100"))

    for _ in range(cfg.confirmation.min_stable):
        counter._reconcile_initial_layers(parsed)

    assert counter._bootstrap_reconciled
    assert counter._template_phase == 1
    assert sum(level == 0 for _cell, level in counter._occupied) == 5
    assert sum(level == 1 for _cell, level in counter._occupied) == 0
    assert {cell for cell, level in counter._occupied if level == 0} == {1, 4, 6, 9, 13}


def test_partial_only_selects_two_levels_when_upper_box_explains_occlusion() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    cfg = load_palletizing_config(config_path)
    counter = GridCounter(
        np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.int32),
        cfg,
    )
    counter.set_box_class("coin_roll_100")
    even = get_layout_template("coin_roll_100", 0)
    odd = get_layout_template("coin_roll_100", 1)
    assert even is not None and odd is not None
    lower = even.slots[5]
    partial = TemplateObservation(
        lower.u,
        lower.v + lower.height * 0.30,
        lower.width,
        lower.height * 0.40,
    )

    upper_observations = (
        *tuple(_observation(even.slots[index]) for index in (2, 7, 10, 13)),
        *tuple(_observation(odd.slots[index]) for index in (0, 3, 6, 11, 14)),
    )
    explained = counter._select_template_bootstrap_fit(
        upper_observations, (partial,), base_level=0,
    )
    unexplained = counter._select_template_bootstrap_fit(
        upper_observations,
        (TemplateObservation(0.05, 0.95, 0.04, 0.04),),
        base_level=0,
    )

    assert explained is not None
    assert explained.phase == 0
    assert {assignment.level for assignment in explained.assignments} == {0, 1}
    assert unexplained is None


def test_empty_start_waits_for_five_boxes_before_locking_template() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    cfg = load_palletizing_config(config_path)
    assert cfg.template_bootstrap_min_boxes == 5
    counter = GridCounter(
        np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.int32),
        cfg,
    )
    counter.set_box_class("coin_roll_100")
    counter._level_footprint[0] = (0.32, 0.15)
    even = get_layout_template("coin_roll_100", 0)
    assert even is not None

    for _ in range(cfg.confirmation.min_stable):
        counter._reconcile_initial_layers([])
    assert not counter._bootstrap_reconciled

    def parsed(slots):
        detections = []
        for slot in slots:
            detections.append(((
                int((slot.u - slot.width / 2) * 1000),
                int((slot.v - slot.height / 2) * 1000),
                int((slot.u + slot.width / 2) * 1000),
                int((slot.v + slot.height / 2) * 1000),
            ), 0.98, "coin_roll_100"))
        return detections

    for _ in range(cfg.confirmation.min_stable):
        counter._reconcile_initial_layers(parsed(even.slots[:4]))
    assert not counter._bootstrap_reconciled
    assert counter.scene_state(0.60).boxes == []

    for _ in range(cfg.confirmation.min_stable):
        counter._reconcile_initial_layers(parsed(even.slots[:5]))
    assert counter._bootstrap_reconciled
    assert counter.total == 5
    assert len(counter.scene_state(0.60).boxes) == 5
    assert {(cell, level) for cell, level in counter._occupied} == {
        (cell, 0) for cell in range(5)
    }


def test_renderer_writes_both_patterns_without_camera_background(tmp_path: Path) -> None:
    outputs = render_layout_templates(
        "coin_roll_100", output_dir=tmp_path, base_pattern="auto",
    )

    assert {path.name for path in outputs} == {
        "coin_roll_100_pattern_A.png",
        "coin_roll_100_pattern_B.png",
        "coin_roll_100_patterns.png",
    }
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0


def test_layout_templates_facade_remains_directly_executable(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = (
        project_root
        / "src"
        / "boxarm"
        / "vision"
        / "palletizing"
        / "layout_templates.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--box-class",
            "coin_roll_100",
            "--base-pattern",
            "B",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        env={
            key: value
            for key, value in os.environ.items()
            if key.upper() != "PYTHONPATH"
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert "Patron activo: N0=B, N1=A" in completed.stdout
    assert (tmp_path / "coin_roll_100_pattern_A.png").exists()
    assert (tmp_path / "coin_roll_100_pattern_B.png").exists()
    assert (tmp_path / "coin_roll_100_patterns.png").exists()


def test_selected_template_phase_survives_state_restore(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "palletizing.yaml"
    cfg = load_palletizing_config(config_path)
    roi = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.int32)
    original = GridCounter(roi, cfg)
    original.set_box_class("coin_roll_100")
    original._template_phase = 1
    template = get_layout_template("coin_roll_100", 0, phase=1)
    assert template is not None
    slot = template.slots[0]
    original._occupied.add((slot.cell, 0))
    original._dynamic_positions[(slot.cell, 0)] = (slot.u, slot.v)
    original._footprint[(slot.cell, 0)] = (slot.width, slot.height)
    original._level_footprint[0] = (max(slot.width, slot.height), min(slot.width, slot.height))
    original.total = original.initial = 1
    path = tmp_path / "state.json"
    path.write_text(json.dumps(original.state_dict(0.60)), encoding="utf-8")

    restored = GridCounter(roi, cfg)
    restored.set_box_class("coin_roll_100")
    restored.load_state(path)

    assert restored._template_phase == 1
    assert get_layout_template("coin_roll_100", 2, restored._template_phase) is template
