# image_pipeline.py

"""
This module is intentionally left with blank stubs where you will implement:
- Image capture from camera
- Vein entry & trajectory detection
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class VeinDetectionResult:
    """
    Logical representation of what you need from the image.
    You can customize this.
    """
    entry_x_px: float
    entry_y_px: float
    traj_angle_deg: float  # e.g., orientation of vein in image
    raw_image: Any         # keep raw image if needed (e.g., numpy array)


def capture_image():
    """
    Capture an image from the camera.

    TODO: implement using your preferred method, for example:
      - OpenCV: cv2.VideoCapture
      - Picamera2
    Should return an image object (e.g., numpy array).
    """
    # Example placeholder:
    # import cv2
    # cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()
    # cap.release()
    # if not ret:
    #     raise RuntimeError("Failed to capture image")
    # return frame
    raise NotImplementedError("capture_image() is not implemented yet.")


def detect_vein_entry_and_trajectory(image) -> VeinDetectionResult:
    """
    Process the captured image to find vein entry point and trajectory.

    TODO: implement your vein segmentation / processing here:
      - Preprocess image
      - Segment veins
      - Compute entry point (x,y) in pixel
      - Compute trajectory angle in degrees

    Should return a VeinDetectionResult.

    For now, this stub raises NotImplementedError.
    """
    # Example: return dummy values for testing the rest of the pipeline:
    # return VeinDetectionResult(
    #     entry_x_px=100.0,
    #     entry_y_px=200.0,
    #     traj_angle_deg=45.0,
    #     raw_image=image
    # )
    raise NotImplementedError("detect_vein_entry_and_trajectory() is not implemented yet.")
