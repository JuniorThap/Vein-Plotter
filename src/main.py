# main.py

import time
import RPi.GPIO as GPIO

from hardware_config import BUTTON_PIN
from stepper_motor import Motion2D
from servo_motor import ServoWithLimit
from image_pipeline import capture_image, detect_vein_entry_and_trajectory
from mapping import map_vein_to_motion


def wait_for_button_press():
    print("Waiting for button press...")
    while GPIO.input(BUTTON_PIN) == GPIO.HIGH:
        time.sleep(0.01)
    print("Button pressed!")
    while GPIO.input(BUTTON_PIN) == GPIO.LOW:
        time.sleep(0.01)


def perform_cycle(motion: Motion2D, servo: ServoWithLimit):
    # 1. Capture image
    img = capture_image()

    # 2. Process image → detect vein entry
    vein = detect_vein_entry_and_trajectory(img)

    # 3. Mapping to real positions
    target = map_vein_to_motion(vein)

    print(f"Target → X={target.x_mm:.2f} mm, Y={target.y_mm:.2f} mm, Servo={target.servo_angle_deg:.1f}°")

    # 4. Two-axis stepping
    motion.move_to(target.x_mm, target.y_mm)

    # 5. Set servo to angle
    servo.set_angle(target.servo_angle_deg)

    # 6. Sweep until limit switch
    servo.sweep_until_limit(direction=1)


def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    motion = Motion2D()
    servo = ServoWithLimit()

    try:
        while True:
            wait_for_button_press()
            try:
                perform_cycle(motion, servo)
            except NotImplementedError as e:
                print("Missing implementation:", e)

    except KeyboardInterrupt:
        print("Exiting…")

    finally:
        servo.cleanup()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
