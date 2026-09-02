from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boxarm.config import load_pipeline_config
from boxarm.runtime.recording import h264_command
from boxarm.runtime.web_recording import browser_base_url, crop_css_box, recording_page_url


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_exposes_three_recording_types() -> None:
    recording = load_pipeline_config(ROOT / "configs" / "pipeline.yaml").recording

    assert recording.types.normal is True
    assert recording.types.iso is True
    assert recording.types.dashboard is True
    assert recording.type_enabled("normal") is True
    assert recording.type_enabled("iso") is True
    assert recording.type_enabled("dashboard") is True
    assert recording.transcode_h264 is True
    assert recording.transcode_timeout_s == 900.0


def test_web_recording_has_explicit_viewport_settings() -> None:
    dashboard = load_pipeline_config(ROOT / "configs" / "pipeline.yaml").recording.dashboard

    assert dashboard.width == 1920
    assert dashboard.height == 900
    assert dashboard.capture_host == "127.0.0.1"
    assert dashboard.azimuth_deg == 55.0
    assert dashboard.elevation_deg == 23.0
    assert 0 <= dashboard.jpeg_quality <= 100


def test_crop_css_box_maps_browser_coordinates_to_screenshot_pixels() -> None:
    frame = np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3)

    crop = crop_css_box(
        frame,
        {"x": 5.0, "y": 3.0, "width": 10.0, "height": 6.0},
        viewport_width=10,
        viewport_height=6,
    )

    np.testing.assert_array_equal(crop, frame[6:12, 10:20])


def test_browser_url_uses_explicit_capture_host() -> None:
    assert browser_base_url("127.0.0.1", 5000) == "http://127.0.0.1:5000"
    assert browser_base_url("192.168.1.25", 5100) == "http://192.168.1.25:5100"
    assert browser_base_url("localhost", 5200) == "http://localhost:5200"
    assert browser_base_url("::1", 5300) == "http://[::1]:5300"
    with pytest.raises(ValueError, match="no es navegable"):
        browser_base_url("0.0.0.0", 5000)


def test_recording_page_url_carries_its_private_view_angle() -> None:
    recording = load_pipeline_config(ROOT / "configs" / "pipeline.yaml").recording

    assert recording_page_url("http://127.0.0.1:5000", 3, recording.dashboard) == (
        "http://127.0.0.1:5000/cam/3?az=55.0&el=23.0"
    )


def test_h264_transcode_command_is_mobile_compatible() -> None:
    command = h264_command("ffmpeg", Path("input.mp4"), Path("output.mp4"))

    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-tag:v") + 1] == "avc1"
    assert command[command.index("-movflags") + 1] == "+faststart"
    assert "scale=trunc(iw/2)*2:trunc(ih/2)*2" in command
