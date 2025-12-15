import pigpio
import time

SERVO_PIN = 18

pi = pigpio.pi()
if not pi.connected:
	exit()

def set_angle(angle):
	pulse_width = 500 + (angle / 180.0) * 2000
	pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)

try:
	while True:
		print("0")
		set_angle(0)
		time.sleep(1)

		print("90")
		set_angle(90)
		time.sleep(1)

		print("180")
		set_angle(180)
		time.sleep(1)
		
		print("270")
		set_angle(270)
		time.sleep(1)

except KeyboardInterrupt:
	pass

pi.set_servo_pulsewidth(SERVO_PIN, 0)
pi.stop()
