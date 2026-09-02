from scripts.calibrate_roi import _camera_id


def test_calibrator_asks_for_camera_when_not_given(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    assert _camera_id(None) == 2


def test_calibrator_keeps_explicit_camera_without_prompt(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(AssertionError()))

    assert _camera_id(3) == 3
