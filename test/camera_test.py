from picamera2 import Picamera2
from libcamera import controls
import RPi.GPIO as GPIO
import cv2
import time
from src.vein_selection import build_model, pipeline

# -----------------------------
# GPIO SETUP
# -----------------------------
IR_PIN = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_PIN, GPIO.OUT)
GPIO.output(IR_PIN, GPIO.LOW)  # IR OFF initially

ir_on = False

# -----------------------------
# CAMERA SETUP
# -----------------------------
print("Camera Setup")
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)

picam2.set_controls({
    "AfMode": controls.AfModeEnum.Manual,
    "LensPosition": 1.0
})

picam2.configure(config)
picam2.start()

print("===================================")
print("CAMERA + IR TEST")
print("Press 'i' : Toggle IR ON / OFF")
print("Press 'q' : Quit")
print("===================================")

model = build_model("", program=True)

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    frame = picam2.capture_array()
    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('i'):
        ir_on = not ir_on
        GPIO.output(IR_PIN, GPIO.HIGH if ir_on else GPIO.LOW)
        print(f"[IR] {'ON' if ir_on else 'OFF'}")
        time.sleep(0.3)  # debounce
    
    if key == ord('c'):
        output = pipeline(model, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), auto=False)
        plot, lines = cv2.imshow("Output")
        cv2.imshow(plot)
        cv2.waitKey(0)


    if key == ord('q'):
        break

# -----------------------------
# CLEANUP
# -----------------------------
cv2.destroyAllWindows()
GPIO.output(IR_PIN, GPIO.LOW)
GPIO.cleanup()
picam2.stop()
