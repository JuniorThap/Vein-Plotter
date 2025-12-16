import pigpio
import time

from src.hardware_config import (
    LIMIT_SWITCH_PIN,
    HOMING_X_LIMIT_SWITCH_PIN,
    HOMING_Y_LIMIT_SWITCH_PIN
)

pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("pigpio daemon not running. Run: sudo pigpiod")

# Configure all pins as INPUT with internal pull-up
for pin in [
    LIMIT_SWITCH_PIN,
    HOMING_X_LIMIT_SWITCH_PIN,
    HOMING_Y_LIMIT_SWITCH_PIN
]:
    pi.set_mode(pin, pigpio.INPUT)
    pi.set_pull_up_down(pin, pigpio.PUD_UP)

print("Reading limit switches (Ctrl+C to exit)")
print("1 = NOT pressed, 0 = PRESSED\n")

try:
    while True:
        z_val = pi.read(LIMIT_SWITCH_PIN)
        x_val = pi.read(HOMING_X_LIMIT_SWITCH_PIN)
        y_val = pi.read(HOMING_Y_LIMIT_SWITCH_PIN)

        print(
            f"Z_AXIS: {z_val} | "
            f"X_HOME: {x_val} | "
            f"Y_HOME: {y_val}"
        )

        time.sleep(0.1)  # 100 ms polling

except KeyboardInterrupt:
    print("\nExit")

finally:
    pi.stop()
