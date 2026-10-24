import pytest

from lfa.detection import (DetectionError, calculate_real_world_distance,
                           check_prediction_classes, roi_slope_center)

from conftest import C1, C2, ONE, FakeYolo, _Box, _Result


def test_roi_geometry_picks_nearest_control():
    midpoint, slope, distance, target = roi_slope_center([C1, C2], [ONE])
    assert midpoint == (100, 250)
    assert target == (450, 250)
    assert distance == pytest.approx(350.57, abs=0.05)
    # C1 and C2 are symmetric about the "1" marker, so only the sign depends on the tie-break
    assert abs(slope) == pytest.approx(20 / 350)


def test_real_world_distance_conversion():
    # 2.55 cm spans 255 px -> 51 px is 0.51 cm
    assert calculate_real_world_distance(255, 51) == pytest.approx(0.51)


def test_check_prediction_classes_ok():
    boxes_c, boxes_1 = check_prediction_classes(_Result(), FakeYolo.names)
    assert len(boxes_c) == 2 and len(boxes_1) == 1


def test_check_prediction_classes_wrong_counts():
    result = _Result()
    result.boxes = result.boxes[:2]  # drop the "1" marker
    with pytest.raises(DetectionError):
        check_prediction_classes(result, FakeYolo.names)

    result.boxes = [_Box(0, C1), _Box(1, ONE)]  # only one control
    with pytest.raises(DetectionError):
        check_prediction_classes(result, FakeYolo.names)
