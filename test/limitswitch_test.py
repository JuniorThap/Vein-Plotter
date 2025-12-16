import pigpio
import time

from src.hardware_config import LIMIT_SWITCH_PIN

pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("pigpio daemon not running. Run: sudo pigpiod")

# Configure pin
pi.set_mode(LIMIT_SWITCH_PIN, pigpio.INPUT)
pi.set_pull_up_down(LIMIT_SWITCH_PIN, pigpio.PUD_UP)

print("Reading limit switch (Ctrl+C to exit)")
print("1 = NOT pressed, 0 = PRESSED")

try:
    while True:
        val = pi.read(LIMIT_SWITCH_PIN)
        print(val)
        time.sleep(0.1)   # read every 100 ms
except KeyboardInterrupt:
    print("\nExit")

finally:
    pi.stop()
