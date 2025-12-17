# stepper_motor.py

import RPi.GPIO as GPIO
import time
from src.hardware_config import (
    STEPPER_X_DIR, STEPPER_X_STEP, MICROSTEP_X,
    STEPPER_Y_DIR, STEPPER_Y_STEP, MICROSTEP_Y,
    STEP_DELAY_SEC, HOMING_X_LIMIT_SWITCH_PIN, HOMING_Y_LIMIT_SWITCH_PIN,
    MOTOR_STEP
)

# GT2
PULLEY_TEETH = 20
BELT_PITCH = 2

class StepperAxis:
    """Represent a single axis (X or Y stepper)."""

    def __init__(self, dir_pin, step_pin, homing_pin, microsteps, motor_steps, reverse=False):
        self.dir_pin = dir_pin
        self.step_pin = step_pin
        self.homing_pin = homing_pin
        self.microsteps = microsteps
        self.motor_steps = motor_steps
        self.position_mm = 0.0
        self.offset = 0.0
        self.reverse = reverse

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(dir_pin, GPIO.OUT)
        GPIO.setup(step_pin, GPIO.OUT)
        GPIO.setup(homing_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def mm2step(self, mm):
        mm_per_rev = PULLEY_TEETH * BELT_PITCH
        steps_per_rev = self.microsteps * self.motor_steps
        steps = mm * (steps_per_rev / mm_per_rev)
        return int(steps)

    def move_to(self, target_mm):
        delta = (target_mm + self.offset) - self.position_mm
        if delta == 0:
            return

        direction = GPIO.HIGH if delta > 0 else GPIO.LOW
        if self.reverse:
            direction = GPIO.LOW if direction == GPIO.HIGH else GPIO.HIGH
        steps = self.mm2step(abs(delta))

        GPIO.output(self.dir_pin, direction)

        for _ in range(steps):
            GPIO.output(self.step_pin, GPIO.HIGH)
            time.sleep(STEP_DELAY_SEC)
            GPIO.output(self.step_pin, GPIO.LOW)
            time.sleep(STEP_DELAY_SEC)

        self.position_mm = target_mm
    
    def homing(self):
        # while GPIO.input(self.homing_pin) == 0:
        #     self.move_to(self.position_mm-1)
        self.position_mm = 0


class Motion2D:
    """Two-axis (X,Y) movement controller."""

    def __init__(self):
        self.x = StepperAxis(STEPPER_X_DIR, STEPPER_X_STEP, HOMING_X_LIMIT_SWITCH_PIN, MICROSTEP_X, MOTOR_STEP)
        self.y = StepperAxis(STEPPER_Y_DIR, STEPPER_Y_STEP, HOMING_Y_LIMIT_SWITCH_PIN, MICROSTEP_Y, MOTOR_STEP, reverse=True)

        self.homing()

    def move_to(self, x_mm, y_mm):
        """Move both axes sequentially (safe for most mechanical systems)."""
        print(f"Moving X → {x_mm:.2f} mm")
        self.x.move_to(x_mm)

        print(f"Moving Y → {y_mm:.2f} mm")
        self.y.move_to(y_mm)

    def get_position(self):
        return self.x.position_mm, self.y.position_mm
    
    def homing(self):
        self.x.homing()
        self.y.homing()
    
    def set_offset(self, offset_x, offset_y):
        self.x.offset = offset_x
        self.y.offset = offset_y

    def move_dir(self, dir_x, dir_y):
        x_mm, y_mm = self.get_position()
        self.move_to(x_mm+dir_x, y_mm+dir_y)