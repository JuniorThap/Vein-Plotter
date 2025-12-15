# main.py

import time
import RPi.GPIO as GPIO

from hardware_config import BUTTON_PIN
from stepper_motor import Motion2D
from servo_motor import ServoWithLimit
from vein_selection import build_model
from image_pipeline import Camera
from mapping import map_vein_to_motion


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

def perform_cycle(motion: Motion2D, servo: ServoWithLimit, camera: Camera):
    img = camera.capture_image()
    vein = camera.detect_vein_points(img)

    target = map_vein_to_motion(vein, index=0)
    print(f"Target → X={target.x_mm:.2f} mm, Y={target.y_mm:.2f} mm, Servo={target.servo_angle_deg:.1f}°")
    motion.move_to(target.x_mm, target.y_mm)
    servo.sweep_until_limit(direction=1)

    servo.set_angle(0)

    target = map_vein_to_motion(vein, index=1)
    print(f"Target → X={target.x_mm:.2f} mm, Y={target.y_mm:.2f} mm, Servo={target.servo_angle_deg:.1f}°")
    motion.move_to(target.x_mm, target.y_mm)
    servo.sweep_until_limit(direction=1)


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    motion = Motion2D()
    servo = ServoWithLimit()
    camera = Camera()

    build_model(r"pretrained_unet_vein.pth")

    try:
        while True:
            wait_for_button_press()
            try:
                perform_cycle(motion, servo, camera)
            except NotImplementedError as e:
                print("Missing implementation:", e)

    except KeyboardInterrupt:
        print("Exiting…")

    finally:
        servo.cleanup()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
