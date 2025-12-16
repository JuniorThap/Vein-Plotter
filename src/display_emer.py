import RPi.GPIO as GPIO
from src.hardware_config import (
    EMERGENCY, YELLOW_LIGHT, GREEN_LIGHT
)

class Display:
    def __init__(self):
        self.emergency = EMERGENCY
        self.yellow_light = YELLOW_LIGHT
        self.green_light = GREEN_LIGHT

        GPIO.setmode(GPIO.BCM)

        # Emergency button (NC or NO → pull-up recommended)
        GPIO.setup(self.emergency, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # Indicator lights
        GPIO.setup(self.yellow_light, GPIO.OUT)
        GPIO.setup(self.green_light, GPIO.OUT)

        # Default state
        self.all_off()

    # -------------------------
    # Light controls
    # -------------------------
    def yellow_on(self):
        GPIO.output(self.yellow_light, GPIO.HIGH)

    def yellow_off(self):
        GPIO.output(self.yellow_light, GPIO.LOW)

    def green_on(self):
        GPIO.output(self.green_light, GPIO.HIGH)

    def green_off(self):
        GPIO.output(self.green_light, GPIO.LOW)

    def all_off(self):
        GPIO.output(self.yellow_light, GPIO.LOW)
        GPIO.output(self.green_light, GPIO.LOW)

    # -------------------------
    # Emergency logic
    # -------------------------
    def is_emergency_pressed(self):
        """
        Emergency button active LOW
        """
        return GPIO.input(self.emergency) == GPIO.LOW

    def update(self):
        """
        Call this in your main loop
        """
        if self.is_emergency_pressed():
            self.green_off()
            self.yellow_on()
        else:
            self.yellow_off()
            self.green_on()

    # -------------------------
    # Cleanup
    # -------------------------
    def cleanup(self):
        self.all_off()
        GPIO.cleanup([
            self.emergency,
            self.yellow_light,
            self.green_light
        ])
