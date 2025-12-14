from picamera2 import Picamera2, Preview
from libcamera import controls
import RPi.GPIO as GPIO
import cv2
import time
import os

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

PERSON_ID = get_next_person_id(folder)
print("Current NEXT_ID:", PERSON_ID)

# Capture sequence per person
CAPTURE_SEQUENCE = [
    ("L1", "NoIR"),
    ("L1", "IR"),
    ("R1", "NoIR"),
    ("R1", "IR"),
]

seq_idx = 0  # 0–3 per person

IR_PIN = 18  # GPIO pin for Relay

# -----------------------------
# GPIO Setup
# -----------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_PIN, GPIO.OUT)
GPIO.output(IR_PIN, GPIO.LOW)  # IR OFF initially

# -----------------------------
# Camera Setup
# -----------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.set_controls({
    "AfMode": controls.AfModeEnum.Manual,
    "LensPosition": 1   # ~15–20 cm (adjust if needed)
})
picam2.configure(config)
picam2.start()

print("=========================================")
print("VEIN SCANNER CONTROL")
print("Press '1' : Capture (L/R × IR/NoIR auto)")
print("Press 'q' : Quit")
print("=========================================")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    frame = picam2.capture_array()
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # -----------------------------
    # CAPTURE STEP
    # -----------------------------
    if key == ord('1'):
        hand, mode = CAPTURE_SEQUENCE[seq_idx]

        if mode == "NoIR":
            GPIO.output(IR_PIN, GPIO.LOW)
            time.sleep(4)

            filename = f"person_{PERSON_ID:03d}_{hand}_NoIR.png"
            cv2.imwrite(os.path.join(folder, filename), frame)

        else:  # IR
            GPIO.output(IR_PIN, GPIO.HIGH)
            time.sleep(4)

            frame_ir = picam2.capture_array()
            filename = f"person_{PERSON_ID:03d}_{hand}_IR.png"
            cv2.imwrite(os.path.join(folder, filename), frame_ir)

            GPIO.output(IR_PIN, GPIO.LOW)

        print(f"[Saved] {filename}")

        seq_idx += 1

        # After 4 captures → next person
        if seq_idx == 4:
            seq_idx = 0
            PERSON_ID += 1
            print(f"=== NEXT PERSON: {PERSON_ID:03d} ===")

        time.sleep(1)

    # -----------------------------
    # QUIT
    # -----------------------------
    if key == ord('q'):
        break

# -----------------------------
# CLEANUP
# -----------------------------
cv2.destroyAllWindows()
GPIO.cleanup()
picam2.stop()
