from src.stepper_motor import Motion2D
from src.servo_motor import ServoWithLimit
from src.mapping import pixel_to_mm, IMG_W, IMG_H, FIELD_X_MM, FIELD_Y_MM
from src.image_pipeline import Camera
import time
import json
import os
import cv2

CALIB_FILE = "calibration_offset.json"

def load_calibration(path=CALIB_FILE):
        if not os.path.exists(path):
            return None

        with open(path, "r") as f:
            return json.load(f)
        
class StepperCalibration:
    def __init__(self, motion: Motion2D, servo: ServoWithLimit, camera: Camera):
        self.motion = motion
        self.servo = servo
        self.camera = camera

    

    def save_calibration(offset_x_mm, offset_y_mm, path=CALIB_FILE):
        data = {
            "offset_x_mm": offset_x_mm,
            "offset_y_mm": offset_y_mm,
            "timestamp": time.time()
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=4)


    def calibrate(self, force=False):

        calib = load_calibration()

        if calib is not None and not force:
            print("[INFO] Calibration already exists. Skipping calibration.")
            self.motion.set_offset(
                calib["offset_x_mm"],
                calib["offset_y_mm"]
            )
            print(
                f"[INFO] Loaded offset X {calib['offset_x_mm']} mm, "
                f"offset Y {calib['offset_y_mm']} mm"
            )
            return

        # Dot at some position
        in_x_px, in_y_px = IMG_W // 2 + 40, IMG_H // 2
        in_x_mm, in_y_mm = pixel_to_mm(in_x_px, IMG_W, FIELD_X_MM), pixel_to_mm(in_y_px, IMG_H, FIELD_Y_MM)
        print("HERERE", in_x_px, in_x_mm)
        self.motion.move_to(in_x_mm, in_y_mm)
        self.servo.sweep_until_limit(direction=1)

        # get back to position
        self.servo.set_angle(0)
        self.motion.homing()

        self.camera.turn_ir_off()
        time.sleep(4)

        img = self.camera.capture_image()
        dots, masked = self.camera.detect_dot(img)
        cv2.imwrite("calibrate_img.png", img)

        if not dots:
            raise RuntimeError("Calibration failed: no dot detected")

        out_x_px, out_y_px = dots[0]
        offset_x_mm = pixel_to_mm(in_x_px-out_x_px)
        offset_y_mm = pixel_to_mm(in_y_px-out_y_px)

        # Apply offset
        self.motion.set_offset(offset_x_mm, offset_y_mm)

        # Save to JSON
        self.save_calibration(offset_x_mm, offset_y_mm)

        print(
            f"[INFO] Calibration complete. "
            f"Offset X {offset_x_mm} mm, Offset Y {offset_y_mm} mm"
        )

        self.camera.turn_ir_on()


if __name__ == "__main__":
    motion = Motion2D()
    servo = ServoWithLimit()
    camera = Camera()
    calibrate = StepperCalibration(motion, servo, camera)

    print("Homing")
    motion.homing()
    print("Start Calibration in 5")
    time.sleep(5)
    print("START")
    calibrate.calibrate(force=True)