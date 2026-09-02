from __future__ import annotations

from types import SimpleNamespace

from boxarm.vision.palletizing import SceneBox, SceneState
from boxarm.web import streaming
from boxarm.web.streaming import _JpegStore, make_flask_app, normal_store, scene_store


def test_mjpeg_transmission_is_capped_without_slowing_producer(monkeypatch) -> None:
    clock = [0.0]

    monkeypatch.setattr(streaming.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(streaming.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    store = _JpegStore("test")
    store.register(7)
    store.write(7, b"primero")
    output = store.mjpeg_gen(7, mjpeg_poll_s=0.01, max_fps=8.0)

    assert b"primero" in next(output)
    store.write(7, b"ultimo")
    assert b"ultimo" in next(output)
    assert clock[0] >= 1.0 / 8.0


def _app():
    cam = SimpleNamespace(id=91, tag="Prueba")
    runtime = SimpleNamespace(mjpeg_poll_s=0.02)
    return make_flask_app((cam,), runtime, (35.0, 35.0))


def test_camera_route_renders_video_and_interactive_iso_together() -> None:
    response = _app().test_client().get("/cam/91")

    assert response.status_code == 200
    assert response.mimetype == "text/html"
    assert b'src="/cam/91/stream"' in response.data
    assert b'id="camera-stream"' in response.data
    assert b'id="iso"' in response.data


def test_camera_route_accepts_a_private_initial_view_for_recording() -> None:
    response = _app().test_client().get("/cam/91?az=55&el=23")

    assert b'data-az0="55.0"' in response.data
    assert b'data-el0="23.0"' in response.data


def test_camera_snapshot_returns_one_finite_latest_jpeg() -> None:
    normal_store.write(91, b"jpeg-prueba")

    response = _app().test_client().get("/cam/91/snapshot")

    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.data == b"jpeg-prueba"
    assert response.headers["Cache-Control"] == "no-store"


def test_old_standalone_iso_view_is_removed() -> None:
    response = _app().test_client().get("/cam/91/iso")

    assert response.status_code == 404


def test_home_has_one_unified_link_per_camera() -> None:
    response = _app().test_client().get("/")

    assert response.status_code == 200
    assert b'class="camera-card"' in response.data
    assert b'/cam/91?view=iso' in response.data
    assert b'/cam/91/iso' not in response.data


def test_camera_status_includes_disabled_and_no_signal_states() -> None:
    enabled = SimpleNamespace(id=191, tag="Activa", enabled=True)
    disabled = SimpleNamespace(id=192, tag="Apagada", enabled=False)
    runtime = SimpleNamespace(mjpeg_poll_s=0.02)
    app = make_flask_app((enabled, disabled), runtime, (35.0, 35.0))

    response = app.test_client().get("/api/cameras")

    assert response.status_code == 200
    payload = {camera["id"]: camera for camera in response.get_json()["cameras"]}
    assert payload[191]["status"] == "no_signal"
    assert payload[192]["status"] == "disabled"


def test_iso_scene_endpoint_returns_latest_geometry_and_rgb_config() -> None:
    cam = SimpleNamespace(id=91, tag="Prueba")
    runtime = SimpleNamespace(mjpeg_poll_s=0.02)
    iso = SimpleNamespace(
        pallet_width_m=1.2,
        pallet_length_m=1.0,
        fill_margin=0.85,
        level_gap_ratio=0.9,
    )
    drawing = SimpleNamespace(
        hud_background=(10, 20, 30),
        color_hud_title=(0, 200, 255),
        color_roi=(1, 2, 3),
        level_colors=((4, 5, 6),),
    )
    app = make_flask_app((cam,), runtime, (35.0, 35.0), iso, drawing)
    scene_store.write(91, SceneState(
        boxes=[SceneBox(7, 1, 0.5, 0.5, 0.2, 0.3, 0.1, 0.1)],
        overlaps=[], level_tops=[0.0, 0.1, 0.2], total_height=0.2,
        total=1, initial=0, placed=1, levels=2,
        provisional_boxes=[
            SceneBox(-1, 1, 0.7, 0.5, 0.1, 0.3, 0.1, 0.1,
                     status="confirming"),
        ],
    ))

    response = app.test_client().get("/cam/91/iso/scene")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["scene"]["boxes"][0]["level"] == 1
    assert payload["scene"]["provisional_boxes"][0]["status"] == "confirming"
    assert payload["view"]["pallet_width"] == 1.2
    assert payload["colors"]["background"] == [30, 20, 10]
    assert response.headers["Cache-Control"] == "no-store"
