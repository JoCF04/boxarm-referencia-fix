from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from boxarm.vision.inference import _with_provisional_boxes
from boxarm.vision.palletizing import SceneBox, SceneState
from boxarm.vision import drawing


def _box(*, status: str, u: float) -> SceneBox:
    return SceneBox(
        cell=-1,
        level=0,
        u=u,
        v=0.5,
        z0=0.0,
        side_a=0.2,
        side_b=0.1,
        height=0.06,
        box_class="coin_roll_100",
        status=status,
    )


def _scene(*, validating_initial: bool, provisional: list[SceneBox]) -> SceneState:
    return SceneState(
        boxes=[],
        overlaps=[],
        level_tops=[0.0, 0.06],
        total_height=0.06,
        total=0,
        initial=0,
        placed=0,
        levels=1,
        provisional_boxes=provisional,
        validating_initial=validating_initial,
    )


def test_raw_observations_replace_internal_hypothesis_during_bootstrap() -> None:
    internal = [_box(status="initializing", u=0.1 + index * 0.01) for index in range(12)]
    observed = [_box(status="confirming", u=0.2 + index * 0.01) for index in range(16)]

    result = _with_provisional_boxes(
        _scene(validating_initial=True, provisional=internal),
        observed,
    )

    assert len(result.provisional_boxes) == 16
    assert {box.status for box in result.provisional_boxes} == {"initializing"}


def test_internal_hypothesis_is_kept_when_bootstrap_has_no_observations() -> None:
    internal = [_box(status="initializing", u=0.4)]

    result = _with_provisional_boxes(
        _scene(validating_initial=True, provisional=internal),
        [],
    )

    assert result.provisional_boxes == internal


def test_tracking_keeps_confirmed_scene_and_appends_confirming_observation() -> None:
    previous = [_box(status="confirming", u=0.3)]
    observed = [_box(status="confirming", u=0.7)]

    result = _with_provisional_boxes(
        _scene(validating_initial=False, provisional=previous),
        observed,
    )

    assert result.provisional_boxes == previous + observed
    assert {box.status for box in result.provisional_boxes} == {"confirming"}


def test_camera_hud_hides_internal_total_while_initial_state_is_pending(monkeypatch) -> None:
    texts: list[str] = []
    monkeypatch.setattr(drawing.cv2, "rectangle", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        drawing.cv2,
        "putText",
        lambda _frame, text, *_args, **_kwargs: texts.append(text),
    )
    cfg = SimpleNamespace(
        hud_width=220, hud_height=220, hud_background=(0, 0, 0),
        hud_title_font_scale=0.5, color_hud_title=(0, 0, 0), hud_title_thickness=1,
        hud_line_font_scale=0.5, color_new=(0, 0, 0), hud_line_thickness=1,
        color_text=(0, 0, 0), hud_visible_thickness=1, hud_fps_font_scale=0.5,
        hud_fps_thickness=1,
    )
    counter = SimpleNamespace(total=12, initial=12, placed=0, visible=16)

    drawing.draw_hud(
        np.zeros((240, 240, 3), dtype=np.uint8),
        "Camara 3",
        counter,
        4.7,
        cfg,
        validating_initial=True,
        observed_count=16,
    )

    assert "Analizando: 16" in texts
    assert not any("En paleta" in text or "inicial 12" in text for text in texts)
