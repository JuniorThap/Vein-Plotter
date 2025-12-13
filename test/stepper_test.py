import RPi.GPIO as GPIO
import time

DIR_PIN = 17
STEP_PIN = 18

STEPS_PER_REV = 200
STEP_DELAY = 0.001

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(DIR_PIN, GPIO.OUT)
GPIO.setup(STEP_PIN, GPIO.OUT)

def step_motor(steps, direction):
	GPIO.output(DIR_PIN, direction)
	for _ in range(steps):
		GPIO.output(STEP_PIN, GPIO.HIGH)
		time.sleep(STEP_DELAY)
		GPIO.output(STEP_PIN, GPIO.LOW)
		time.sleep(STEP_DELAY)

try:
	while True:
		print("Rotate CW")
		step_motor(STEPS_PER_REV, GPIO.HIGH)
		time.sleep(1)

		print("Rotate CCW")
		step_motor(STEPS_PER_REV, GPIO.LOW)
		time.sleep(1)

except KeyboardInterrupt:
	print("Stopping motor")

finally:
	GPIO.cleanup()
