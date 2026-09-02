import cv2
import numpy as np

from boxarm.vision.inference import _bbox_fully_inside_roi


def test_bbox_allows_small_edge_overrun_but_not_large_overrun() -> None:
    roi = np.array([[10, 10], [10, 90], [90, 90], [90, 10]], dtype=np.float32)

    assert _bbox_fully_inside_roi((8, 20, 50, 50), roi, tolerance_px=3)
    assert not _bbox_fully_inside_roi((5, 20, 50, 50), roi, tolerance_px=3)
