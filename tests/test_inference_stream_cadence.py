"""Regresiones del ritmo del stream normal de cámara."""

import inspect

from boxarm.vision import inference


def test_each_inference_iteration_publishes_one_annotated_jpeg() -> None:
    """No alternar un preview crudo con el frame final del mismo instante."""
    source = inspect.getsource(inference.run_inference)

    assert source.count("push_frame(jpeg_out, buf.tobytes())") == 1
