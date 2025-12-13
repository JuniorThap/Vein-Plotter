# servo_motor.py

import time
import RPi.GPIO as GPIO
from hardware_config import (
    SERVO_PIN,
    SERVO_FREQ_HZ,
    SERVO_MIN_DUTY,
    SERVO_MAX_DUTY,
    LIMIT_SWITCH_PIN,
)


class ServoWithLimit:
    """
    Simple servo driver using RPi.GPIO PWM + limit switch.

    Limit switch is assumed:
      - Active LOW (pressed = 0)
      - Connected to LIMIT_SWITCH_PIN with pull-up
    """

    def __init__(self):
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        GPIO.setup(LIMIT_SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        self.pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ_HZ)
        self.pwm.start(0)

        self.min_duty = SERVO_MIN_DUTY
        self.max_duty = SERVO_MAX_DUTY
        self.current_angle = 0.0

    def angle_to_duty(self, angle_deg: float) -> float:
        angle_clamped = max(0.0, min(180.0, angle_deg))
        duty = self.min_duty + (angle_clamped / 180.0) * (self.max_duty - self.min_duty)
        return duty

    def set_angle(self, angle_deg: float, settle_time: float = 0.3):
        """
        Move servo to a specific angle (no limit-check here).
        """
        duty = self.angle_to_duty(angle_deg)
        self.pwm.ChangeDutyCycle(duty)
        time.sleep(settle_time)
        self.pwm.ChangeDutyCycle(0.0)  # stop sending continuous pulses (optional)
        self.current_angle = angle_deg

    def sweep_until_limit(
        self,
        direction: int = 1,
        step_deg: float = 2.0,
        step_delay: float = 0.05,
    ):
        """
        Move servo in small steps in `direction` until limit switch is hit.
        direction: +1 or -1
        """
        angle = self.current_angle

        while 0.0 <= angle <= 180.0:
            # Check limit switch (active LOW)
            if GPIO.input(LIMIT_SWITCH_PIN) == 0:
                print("Limit switch triggered, stopping servo.")
                break

            angle += direction * step_deg
            angle = max(0.0, min(180.0, angle))

            duty = self.angle_to_duty(angle)
            self.pwm.ChangeDutyCycle(duty)
            time.sleep(step_delay)

        self.pwm.ChangeDutyCycle(0.0)
        self.current_angle = angle

    def cleanup(self):
        self.pwm.stop()
