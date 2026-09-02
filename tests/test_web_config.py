from pathlib import Path

import pytest

from boxarm.config import ConfigError, load_web_config


def _write_web_config(path: Path, *, stream_max_fps: float) -> None:
    path.write_text(
        "flask_host: '0.0.0.0'\n"
        "port: 8080\n"
        "jpeg_quality: 55\n"
        f"stream_max_fps: {stream_max_fps}\n",
        encoding="utf-8",
    )


def test_web_config_loads_transmission_fps_limit(tmp_path: Path) -> None:
    path = tmp_path / "web.yaml"
    _write_web_config(path, stream_max_fps=8)

    assert load_web_config(path).stream_max_fps == 8.0


@pytest.mark.parametrize("value", [0, -1])
def test_web_config_rejects_non_positive_transmission_fps(tmp_path: Path, value: float) -> None:
    path = tmp_path / "web.yaml"
    _write_web_config(path, stream_max_fps=value)

    with pytest.raises(ConfigError, match="stream_max_fps"):
        load_web_config(path)
