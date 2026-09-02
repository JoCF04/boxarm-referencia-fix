from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

from boxarm.vision.inference import (
    _active_box_class,
    _frame_detections,
    _model_class_ids,
)


ROI = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.int32)


def test_class_names_resolve_against_the_model_regardless_of_id_order() -> None:
    arm_class_id, box_class_ids = _model_class_ids(
        {0: "manipulador", 1: "caja_1sol", 2: "caja_2soles"},
        "manipulador",
        ("caja_1sol", "caja_2soles"),
    )

    assert arm_class_id == 0
    assert box_class_ids == {1, 2}


def test_same_names_survive_a_retrain_that_shifts_every_id() -> None:
    """Reentrenar corre los ids; los nombres configurados no cambian.

    Este es el bug que motivo pasar de ids a nombres: con `arm_class_id: 0`
    el brazo pasaba a ser una caja tras reentrenar, sin ningun error.
    """
    arm_class_id, box_class_ids = _model_class_ids(
        {0: "caja_0.50", 1: "caja_1sol", 2: "caja_2soles", 3: "manipulador"},
        "manipulador",
        ("caja_1sol", "caja_2soles"),
    )

    assert arm_class_id == 3
    assert box_class_ids == {1, 2}


def test_a_class_name_absent_from_the_model_fails_loudly() -> None:
    with pytest.raises(ValueError, match="no existen en el modelo"):
        _model_class_ids(
            {0: "manipulador", 1: "caja_1sol"},
            "manipulador",
            ("caja_1sol", "caja_que_no_se_etiqueto"),
        )


def test_active_box_class_comes_from_config_not_model_order() -> None:
    class_id, class_name = _active_box_class(
        {0: "brazo", 1: "caja_bolsa_0.10", 2: "caja_cartucho_1.00"},
        "caja_cartucho_1.00",
        {"caja_bolsa_0.10": 25, "caja_cartucho_1.00": 15},
    )

    assert (class_id, class_name) == (2, "caja_cartucho_1.00")


def test_current_pallet_capacity_matches_the_exact_model_class() -> None:
    config = yaml.safe_load(Path("configs/palletizing.yaml").read_text(encoding="utf-8"))
    dataset = yaml.safe_load(
        Path("datasets/cnm_palletscajas/data.yaml").read_text(encoding="utf-8")
    )

    assert config["active_box_class"] == dataset["names"][1] == "caja_1sol"
    assert config["boxes_per_level"] == {
        "caja_bolsa_0.10": 25,
        "caja_cartucho_0.10": 25,
        "caja_1sol": 15,
        "caja_cartucho_2.00": 25,
    }
    assert config["gate"]["motion_stable_frames"] == 3
    assert config["confirmation"]["min_stable"] == 3


def test_active_box_class_error_lists_model_names() -> None:
    with pytest.raises(ValueError, match="disponibles=.*caja_1sol"):
        _active_box_class(
            {0: "brazo", 1: "caja_1sol"},
            "caja_cartucho_1.00",
            {"caja_cartucho_1.00": 15},
        )


@dataclass
class FakeBox:
    class_id: int
    bbox: tuple[int, int, int, int]
    confidence: float = 0.9

    @property
    def cls(self) -> np.ndarray:
        return np.array([self.class_id])

    @property
    def xyxy(self) -> np.ndarray:
        return np.array([self.bbox])

    @property
    def conf(self) -> np.ndarray:
        return np.array([self.confidence])


def test_robot_anywhere_in_frame_discards_every_box_class() -> None:
    arm_outside_roi = FakeBox(99, (10, 10, 90, 90))
    box_type_inside_roi = FakeBox(7, (140, 140, 220, 220))

    arm_bboxes, detections = _frame_detections(
        [box_type_inside_roi, arm_outside_roi], arm_class_id=99,
        box_class_ids={3, 7}, roi_pts=ROI,
    )

    assert arm_bboxes == [(10, 10, 90, 90)]
    assert detections == []


def test_without_robot_accepts_all_box_classes_inside_roi() -> None:
    arm_bboxes, detections = _frame_detections(
        [FakeBox(3, (110, 110, 150, 150)), FakeBox(7, (200, 200, 250, 250))],
        arm_class_id=99, box_class_ids={3, 7}, roi_pts=ROI,
    )

    assert arm_bboxes == []
    assert detections == [
        (110, 110, 150, 150, 0.9),
        (200, 200, 250, 250, 0.9),
    ]


def test_without_robot_excludes_non_box_classes_and_boxes_outside_roi() -> None:
    arm_bboxes, detections = _frame_detections(
        [FakeBox(3, (10, 10, 90, 90)), FakeBox(42, (140, 140, 220, 220))],
        arm_class_id=99, box_class_ids={3, 7}, roi_pts=ROI,
    )

    assert arm_bboxes == []
    assert detections == []
