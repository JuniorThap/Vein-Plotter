# mapping.py

from dataclasses import dataclass
from image_pipeline import VeinDetectionResult


@dataclass
class MotionTarget:
    x_mm: float
    y_mm: float
    servo_angle_deg: float


def pixel_to_mm(px, image_px, field_mm):
    """Simple scale conversion placeholder."""
    mm_per_px = field_mm / image_px
    return px * mm_per_px


def map_vein_to_motion(vein_result: VeinDetectionResult) -> MotionTarget:
    """
    Convert image result → machine coordinates.
    Replace this with real calibration later.
    """

    IMG_W = 640
    IMG_H = 480
    FIELD_X_MM = 80
    FIELD_Y_MM = 60

    x_mm = pixel_to_mm(vein_result.entry_x_px, IMG_W, FIELD_X_MM)
    y_mm = pixel_to_mm(vein_result.entry_y_px, IMG_H, FIELD_Y_MM)

    return MotionTarget(
        x_mm=x_mm,
        y_mm=y_mm,
        servo_angle_deg=vein_result.traj_angle_deg
    )
