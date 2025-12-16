# mapping.py

from dataclasses import dataclass
from src.image_pipeline import VeinDetectionResult


IMG_W = 640
IMG_H = 480

MAX_FIELD_X_MM = 205
MAX_FIELD_Y_MM = MAX_FIELD_X_MM * (IMG_H / IMG_W) # 155

CAMERA_HIGH = 300
HAND_HIGH = 80 # Estimatation

FIELD_X_MM = (MAX_FIELD_X_MM / 2) * (CAMERA_HIGH - HAND_HIGH)
FIELD_Y_MM = (MAX_FIELD_Y_MM / 2) * (CAMERA_HIGH - HAND_HIGH)

@dataclass
class MotionTarget:
    x_mm: float
    y_mm: float


def pixel_to_mm(px, image_px, field_mm):
    """Simple scale conversion placeholder."""
    mm_per_px = field_mm / image_px
    return px * mm_per_px


def map_vein_to_motion(vein_result: VeinDetectionResult, index) -> MotionTarget:
    """
    Convert image result → machine coordinates.
    Replace this with real calibration later.
    """

    x_mm = pixel_to_mm(vein_result.points_px[index][0], IMG_W, FIELD_X_MM)
    y_mm = pixel_to_mm(vein_result.points_px[index][1], IMG_H, FIELD_Y_MM)

    return MotionTarget(
        x_mm=x_mm,
        y_mm=y_mm,
    )
