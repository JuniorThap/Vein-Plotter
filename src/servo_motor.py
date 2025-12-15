# servo_motor.py

import time
import pigpio
from src.hardware_config import (
    SERVO_PIN,
    SERVO_MIN_DUTY,
    SERVO_MAX_DUTY,
    LIMIT_SWITCH_PIN,
)


class ServoWithLimit:
    """
    Servo controller using pigpio for stable PWM + limit switch protection.
    
    pigpio sends pulses by microseconds (500–2500 typically), which is
    far more accurate and stable than RPi.GPIO PWM.
    """

    def __init__(self):
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpio daemon is not running")

        # Limit switch
        self.pi.set_mode(LIMIT_SWITCH_PIN, pigpio.INPUT)
        self.pi.set_pull_up_down(LIMIT_SWITCH_PIN, pigpio.PUD_UP)

        # Servo pin
        self.pi.set_mode(SERVO_PIN, pigpio.OUTPUT)

        # Convert duty-cycle style config → microseconds
        # 50 Hz = 20 ms → 2.5% = 500 μs, 12.5% = 2500 μs
        self.min_us = int(20000 * (SERVO_MIN_DUTY / 100.0))
        self.max_us = int(20000 * (SERVO_MAX_DUTY / 100.0))

        self.current_angle = 0.0
        self.set_angle(0)  # initialize

    def angle_to_pulse(self, angle_deg: float) -> int:
        angle_clamped = max(0.0, min(180.0, angle_deg))
        return int(self.min_us + (angle_clamped / 180.0) * (self.max_us - self.min_us))

    def set_angle(self, angle_deg: float, settle_time: float = 0.3):
        """
        Move to a specific angle (blocking).
        pigpio: use set_servo_pulsewidth(pin, microseconds)
        """
        pulse = self.angle_to_pulse(angle_deg)
        self.pi.set_servo_pulsewidth(SERVO_PIN, pulse)
        time.sleep(settle_time)

        # keep pulse ON → servos like continuous pulses
        # (we do NOT stop pulse as in RPi.GPIO version)
        self.current_angle = angle_deg

    def sweep_until_limit(
        self,
        direction: int = 1,
        step_deg: float = 2.0,
        step_delay: float = 0.05,
    ):
        """
        Sweep servo in direction until limit switch is hit.
        direction: +1 = increase angle, -1 = decrease angle
        """

        angle = self.current_angle

        while 0.0 <= angle <= 180.0:
            # check limit switch (active LOW)
            if self.pi.read(LIMIT_SWITCH_PIN) == 0:
                print("Limit switch triggered → stopping servo.")
                break

            angle += direction * step_deg
            angle = max(0.0, min(180.0, angle))

            pulse = self.angle_to_pulse(angle)
            self.pi.set_servo_pulsewidth(SERVO_PIN, pulse)
            time.sleep(step_delay)

        self.current_angle = angle

    def cleanup(self):
        # stop sending pulses
        self.pi.set_servo_pulsewidth(SERVO_PIN, 0)
        self.pi.stop()
