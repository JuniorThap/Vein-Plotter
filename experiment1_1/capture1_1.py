from picamera2 import Picamera2, Preview
from libcamera import controls
import RPi.GPIO as GPIO
import cv2
import time
import os
from datetime import datetime

# -----------------------------
# CREATE EXPERIMENT FOLDER
# -----------------------------
folder = "experiment"
os.makedirs(folder, exist_ok=True)

print(f"Saving images to folder: {folder}")


# -----------------------------
# USER SETTINGS
# -----------------------------
def get_next_person_id(folder):
    max_id = 0
    for f in os.listdir(folder):
        if f.startswith("person_") and f[7:10].isdigit():
            pid = int(f[7:10])
            max_id = max(max_id, pid)
    return max_id + 1

counter = 0
PERSON_ID = get_next_person_id(folder)
HAND_LABEL = ["L1", "R1"]   # L1, R1, etc.

IR_PIN = 17         # GPIO pin for Relay


# -----------------------------
# GPIO Setup
# -----------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_PIN, GPIO.OUT)
GPIO.output(IR_PIN, GPIO.HIGH)   # Relay OFF initially

# -----------------------------
# Camera Setup
# -----------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 1})   # Manual Foscus about 15cm
picam2.configure(config)
picam2.start()

print("=========================================")
print("VEIN SCANNER CONTROL")
print("Press \"SPACEBAR\" : Capture NoIR (IR OFF) and IR (IR ON)")
print("Press 'q' : Quit")
print("=========================================")

while True:
    frame = picam2.capture_array()
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # -----------------------------
    # 1. Capture WITHOUT IR
    # -----------------------------
    if key == ord('1'):
        GPIO.output(IR_PIN, GPIO.HIGH)  # IR OFF

        filename = f"person_{PERSON_ID:03d}_{HAND_LABEL[counter%2]}_NoIR.png"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, frame)

        print(f"[Saved] {filepath}")

    # -----------------------------
    # 2. Capture WITH IR
    # -----------------------------
    if key == ord('2'):
        GPIO.output(IR_PIN, GPIO.LOW)   # IR ON

        frame_ir = picam2.capture_array()
        filename = f"person_{PERSON_ID:03d}_{HAND_LABEL[counter%2]}_IR.png"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, frame_ir)

        print(f"[Saved] {filepath}")

        GPIO.output(IR_PIN, GPIO.HIGH)  # Turn IR OFF after capture
        
    # -----------------------------
    # 3. Update Counter
    # -----------------------------
        counter += 1
        if counter % 2 == 0:
            PERSON_ID += 1
    time.sleep(1)

    # -----------------------------
    # Quit Program
    # -----------------------------
    if key == ord('q'):
        break

cv2.destroyAllWindows()
GPIO.cleanup()
picam2.stop()

