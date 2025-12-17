# image_pipeline.py


from dataclasses import dataclass
from typing import Any, List, Tuple
from picamera2 import Picamera2, Preview
from libcamera import controls
import RPi.GPIO as GPIO
import cv2
import time
import os
from src.vein_selection import pipeline
from src.hardware_config import IR_PIN
from src.mapping import IMG_H, IMG_W
import numpy as np


Point = Tuple[float, float]

@dataclass
class VeinDetectionResult:
    points_px: List[Point]   # [(x, y), (x, y), ...]
    image: Any

class Camera():

    def __init__(self):
        self.ir_pin = IR_PIN
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        self.picam2.set_controls({
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": 1   # ~15–20 cm (adjust if needed)
        })
        self.picam2.configure(config)
        self.picam2.start()

        GPIO.setup(IR_PIN, GPIO.OUT)
        GPIO.output(IR_PIN, GPIO.HIGH)


    def capture_image(self, show=False):
        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if show:
            cv2.imshow("Camera", frame)
            cv2.waitKey(1)

        return frame

    def detect_vein_points(self, model, image) -> VeinDetectionResult:
        img, lines = pipeline(model, image)
        if lines is not None:
            pts = lines[0]["points"]
            return VeinDetectionResult([(pts[0][0], pts[0][1]), (pts[-1][0], pts[-1][1])], img)
        else:
            return VeinDetectionResult([(IMG_W // 2 - 10, IMG_H // 2 - 10), (IMG_W // 2 + 10, IMG_H // 2 + 10)], img)
    
    def detect_dot(self, image, lower=[114, 90, 3], upper=[134, 170, 83]):
        lower = np.array(lower)
        upper = np.array(upper)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers = []

        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            centers.append((cx, cy))

        masked = cv2.bitwise_and(image, image, mask=mask)

        return centers, masked

    def turn_ir_on(self):
        GPIO.output(self.ir_pin, GPIO.HIGH)
    
    def turn_ir_off(self):
        GPIO.output(self.ir_pin, GPIO.LOW)

        
