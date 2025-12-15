# main.py

import time
import RPi.GPIO as GPIO
from hardware_config import BUTTON_PIN
from stepper_motor import Motion2D
from servo_motor import ServoWithLimit
from vein_selection import build_model
from image_pipeline import Camera
from mapping import map_vein_to_motion
import cv2
import os


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

def perform_cycle(motion: Motion2D, servo: ServoWithLimit, camera: Camera, save_dir, person_id, side):
    camera.turn_ir_on()
    time.sleep(4)
    ir_img = camera.capture_image()
    vein = camera.detect_vein_points(ir_img)

    target = map_vein_to_motion(vein, index=0)
    print(f"Target → X={target.x_mm:.2f} mm, Y={target.y_mm:.2f} mm, Servo={target.servo_angle_deg:.1f}°")
    motion.move_to(target.x_mm, target.y_mm)
    servo.sweep_until_limit(direction=1)
    time.sleep(1)
    servo.set_angle(0)


    target = map_vein_to_motion(vein, index=1)
    print(f"Target → X={target.x_mm:.2f} mm, Y={target.y_mm:.2f} mm, Servo={target.servo_angle_deg:.1f}°")
    motion.move_to(target.x_mm, target.y_mm)
    servo.sweep_until_limit(direction=1)
    time.sleep(1)
    servo.set_angle(0)

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

    build_model(r"pretrained_unet_vein.pth")

    folder = "experiment2"
    os.makedirs(folder, exist_ok=True)
    print(f"Saving images to folder: {folder}")

    try:
        while True:
            PERSON_ID = get_next_person_id(folder)
            print(f"\n=== Ready for PERSON {PERSON_ID:03d} ===")

            # ---- L1 ----
            wait_for_button_press()
            perform_cycle(motion, servo, camera, folder, PERSON_ID, "L1")
            
            # ---- R1 ----
            wait_for_button_press()
            perform_cycle(motion, servo, camera, folder, PERSON_ID, "R1")

            print(f"Completed PERSON {PERSON_ID:03d}")

    except KeyboardInterrupt:
        print("Exiting…")

    finally:
        servo.cleanup()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
