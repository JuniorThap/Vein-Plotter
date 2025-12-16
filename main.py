# main.py

import time
import RPi.GPIO as GPIO
from src.hardware_config import BUTTON_PIN
from src.stepper_motor import Motion2D
from src.servo_motor import ServoWithLimit
from src.vein_selection import build_model
from src.image_pipeline import Camera
from src.stepper_calibration import StepperCalibration
from src.display_emer import Display
from src.mapping import map_vein_to_motion
import cv2
import os
import torch


def wait_for_button_press():
    print("Waiting for button press...")
    while GPIO.input(BUTTON_PIN) == GPIO.HIGH:
        time.sleep(0.01)
    print("Button pressed!")
    while GPIO.input(BUTTON_PIN) == GPIO.LOW:
        time.sleep(0.01)

def reset(motion: Motion2D, servo: ServoWithLimit):
    servo.set_angle(0)
    motion.homing()

def perform_cycle(motion: Motion2D, servo: ServoWithLimit, camera: Camera, model, save_dir, person_id, side):
    print("[-----START NEW CYCLE-----]")
    print("[1. TURN IR ON]")
    camera.turn_ir_on()
    time.sleep(4)
    print("[2. CAPTURE IR IMAGE]")
    ir_img = camera.capture_image()
    vein = camera.detect_vein_points(model, ir_img)

    print("[3. MAP TO MOTION]")
    target = map_vein_to_motion(vein, index=0)
    print(f"Target → X={target.x_mm:.2f} mm, Y={target.y_mm:.2f} mm")
    
    print("[4. START MOVING]")
    motion.move_to(target.x_mm, target.y_mm)
    print("[5. DOT THE PEN]")
    servo.sweep_until_limit(direction=1)
    time.sleep(1)
    print("[6. RETURN]")
    servo.set_angle(0)

    print("[7. MAP TO MOTION (NEXT DOT)]")
    target = map_vein_to_motion(vein, index=1)
    print(f"Target → X={target.x_mm:.2f} mm, Y={target.y_mm:.2f} mm")
    
    print("[8. START MOVING]")
    motion.move_to(target.x_mm, target.y_mm)
    print("[9. DOT THE PEN]")
    servo.sweep_until_limit(direction=1)
    time.sleep(1)
    print("[10. RETURN]")
    servo.set_angle(0)


    print("[11. CLOSE IR AND HOMING]")
    camera.turn_ir_off()
    time.sleep(4)
    motion.homing()

    img = camera.capture_image()

    if save_dir is not None:
        ir_path = os.path.join(
            save_dir, f"person_{person_id:03d}_{side}_IR.png"
        )
        noir_path = os.path.join(
            save_dir, f"person_{person_id:03d}_{side}_NoIR.png"
        )

        cv2.imwrite(ir_path, ir_img)
        cv2.imwrite(noir_path, img)

        print(f"Saved {ir_path}")
        print(f"Saved {noir_path}")

    print("[-----END CYCLE-----]")


def get_next_person_id(folder):
        max_id = 0
        for f in os.listdir(folder):
            if f.startswith("person_") and f[7:10].isdigit():
                pid = int(f[7:10])
                max_id = max(max_id, pid)
        return max_id + 1


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    motion = Motion2D()
    servo = ServoWithLimit()
    camera = Camera()
    calibrate = StepperCalibration(motion, servo, camera)
    display = Display()

    display.green_on()
    model = build_model(r"pretrained_unet_vein.p2", program=True)

    folder = "experiment2"
    os.makedirs(folder, exist_ok=True)
    print(f"Saving images to folder: {folder}")

    time.sleep(5)
    print("Start Calibration")
    display.yellow_on()
    calibrate.calibrate(force=False)
    display.yellow_off()

    servo.set_angle(0)
    motion.homing()

    try:
        while True:
            PERSON_ID = get_next_person_id(folder)
            print(f"\n=== Ready for PERSON {PERSON_ID:03d} ===")

            # ---- L1 ----
            wait_for_button_press()
            display.yellow_on()
            perform_cycle(motion, servo, camera, model, folder, PERSON_ID, "L1")
            display.yellow_off()
            
            # ---- R1 ----
            wait_for_button_press()
            display.yellow_on()
            perform_cycle(motion, servo, camera, model, folder, PERSON_ID, "R1")
            display.yellow_off()

            print(f"Completed PERSON {PERSON_ID:03d}")

    except KeyboardInterrupt:
        print("Exiting…")
        display.green_off()

    finally:
        servo.cleanup()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
