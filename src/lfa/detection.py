"""YOLO detection of strip markers and strip geometry.

The YOLO model detects three kinds of markers on each strip image:
two control markers (class ``C``) and one signal marker (class ``1``).
From their positions we derive the sampling line along the strip and the
pixel length of the known real-world distance (default 2.55 cm).
"""

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DetectionError(Exception):
    """Raised when the expected markers are not found in an image."""


def load_yolo(weights_path):
    """Load YOLO weights. Returns ``(model, class_names)``."""
    from ultralytics import YOLO  # imported lazily so the rest of the package works without it

    model = YOLO(str(weights_path))
    return model, model.names


def return_box_location(cls, boxes, class_names):
    """Return the ``xywh`` boxes whose predicted class name equals ``cls``."""
    class_location = []
    for box in boxes:
        class_id = int(box.cls[0].item())
        if class_names[class_id] == cls:
            class_location.append(box.xywh[0].tolist())
    return class_location


def check_prediction_classes(prediction, class_names):
    """Require exactly two control (``C``) markers and one ``1`` marker."""
    boxes = prediction.boxes
    boxes_c = return_box_location("C", boxes, class_names)
    boxes_1 = return_box_location("1", boxes, class_names)
    if len(boxes_c) != 2:
        raise DetectionError("Incorrect Number of Control Signal ID")
    if len(boxes_1) != 1:
        raise DetectionError("Incorrect Number of 1 Signal ID")
    return boxes_c, boxes_1


def roi_slope_center(boxes_c, boxes_1):
    """Derive the sampling geometry from marker positions.

    Returns
    -------
    midpoint : (x, y) midpoint between the two control markers
    slope    : slope of the line from the nearest control marker to the "1" marker
    distance : pixel distance between that control marker and the "1" marker
    target   : (x, y) of the "1" marker
    """
    c1 = boxes_c[0]
    c2 = boxes_c[1]
    midpt_c = (c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2

    dist_c1_1 = np.sqrt((c1[0] - boxes_1[0][0]) ** 2 + (c1[1] - boxes_1[0][1]) ** 2)
    slope_c1_1 = (c1[1] - boxes_1[0][1]) / (c1[0] - boxes_1[0][0])
    dist_c2_1 = np.sqrt((c2[0] - boxes_1[0][0]) ** 2 + (c2[1] - boxes_1[0][1]) ** 2)
    slope_c2_1 = (c2[1] - boxes_1[0][1]) / (c2[0] - boxes_1[0][0])

    target = (boxes_1[0][0], boxes_1[0][1])
    if dist_c1_1 < dist_c2_1:
        return midpt_c, slope_c1_1, dist_c1_1, target
    return midpt_c, slope_c2_1, dist_c2_1, target


def locate_strip(prediction, class_names):
    """Convenience wrapper: YOLO result -> ``(midpoint, slope, distance, target)``."""
    boxes_c, boxes_1 = check_prediction_classes(prediction, class_names)
    return roi_slope_center(boxes_c, boxes_1)


def calculate_real_world_distance(px_distance, peak, rw_distance=2.55):
    """Convert a pixel offset ``peak`` to centimetres.

    ``px_distance`` is the pixel length that corresponds to ``rw_distance`` cm.
    """
    cm_per_pixel = rw_distance / px_distance
    return cm_per_pixel * peak
