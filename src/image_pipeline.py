# image_pipeline.py


from dataclasses import dataclass
from typing import Any
from picamera2 import Picamera2, Preview
from libcamera import controls
import RPi.GPIO as GPIO
import cv2
import time
import os
from vein_selection import build_model, pipeline


@dataclass
class VeinDetectionResult:
    """
    Logical representation of what you need from the image.
    You can customize this.
    """
    x1_px: float
    y1_px: float
    x2_px: float
    y2_px: float
    image: Any


def camera_setup():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.set_controls({
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": 1   # ~15–20 cm (adjust if needed)
    })
    picam2.configure(config)
    picam2.start()

    return picam2

def capture_image(picam2, show=False):
    frame = picam2.capture_array()

    if show:
        cv2.imshow("Camera", frame)
        cv2.waitKey(1)

    return frame



def detect_vein_entry_and_trajectory(model, image) -> VeinDetectionResult:
    img, lines = pipeline(model, image)
    pts = lines[0]["points"]
    return VeinDetectionResult(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1], img)
