# stepper_motor.py

import RPi.GPIO as GPIO
import time
from hardware_config import (
    STEPPER_X_DIR, STEPPER_X_STEP, STEPPER_X_EN, STEPS_PER_MM_X,
    STEPPER_Y_DIR, STEPPER_Y_STEP, STEPPER_Y_EN, STEPS_PER_MM_Y,
    STEP_DELAY_SEC
)


class StepperAxis:
    """Represent a single axis (X or Y stepper)."""

    def __init__(self, dir_pin, step_pin, en_pin, steps_per_mm):
        self.dir_pin = dir_pin
        self.step_pin = step_pin
        self.en_pin = en_pin
        self.steps_per_mm = steps_per_mm
        self.position_mm = 0.0

        GPIO.setup(dir_pin, GPIO.OUT)
        GPIO.setup(step_pin, GPIO.OUT)
        GPIO.setup(en_pin, GPIO.OUT)

        GPIO.output(en_pin, GPIO.HIGH)  # disable initially

    def enable(self):
        GPIO.output(self.en_pin, GPIO.LOW)

    def disable(self):
        GPIO.output(self.en_pin, GPIO.HIGH)

    def move_to(self, target_mm):
        delta = target_mm - self.position_mm
        if delta == 0:
            return

        direction = GPIO.HIGH if delta > 0 else GPIO.LOW
        steps = int(abs(delta) * self.steps_per_mm)

        self.enable()
        GPIO.output(self.dir_pin, direction)

        for _ in range(steps):
            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(STEP_DELAY_SEC)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(STEP_DELAY_SEC)

        self.disable()
        self.position_mm = target_mm


class Motion2D:
    """Two-axis (X,Y) movement controller."""

    def __init__(self):
        self.x = StepperAxis(STEPPER_X_DIR, STEPPER_X_STEP, STEPPER_X_EN, STEPS_PER_MM_X)
        self.y = StepperAxis(STEPPER_Y_DIR, STEPPER_Y_STEP, STEPPER_Y_EN, STEPS_PER_MM_Y)

    def move_to(self, x_mm, y_mm):
        """Move both axes sequentially (safe for most mechanical systems)."""
        print(f"Moving X → {x_mm:.2f} mm")
        self.x.move_to(x_mm)

        print(f"Moving Y → {y_mm:.2f} mm")
        self.y.move_to(y_mm)

    def get_position(self):
        return self.x.position_mm, self.y.position_mm
