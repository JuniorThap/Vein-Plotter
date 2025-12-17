from src.stepper_motor import Motion2D
from src.servo_motor import ServoWithLimit
from src.image_pipeline import Camera
from src.stepper_calibration import save_calibration
from src.vein_selection import build_model
from src.mapping import map_vein_to_motion
import cv2
import numpy as np
import time

cv2.namedWindow("control", cv2.WINDOW_NORMAL)
cv2.imshow("control", np.zeros((100, 300), dtype=np.uint8))

print("Click the window. WASD to move, QE for servo, Z to quit")

motion = Motion2D()
servo = ServoWithLimit()
camera = Camera()

model = build_model("", program=True)


while True:
    img = camera.capture_image()
    cv2.imshow("Camera", img)
    key = cv2.waitKey(1) & 0xFF

    dx, dy = 0, 0

    if key == ord('w'):
        print('w')
        dy = -1
    elif key == ord('s'):
        print('s')
        dy = 1
    elif key == ord('a'):
        print('a')
        dx = -1
    elif key == ord('d'):
        print('d')
        dx = 1
    elif key == ord('q'):
        print('q')
        servo.set_angle(0)
    elif key == ord('e'):
        print('e')
        servo.sweep_until_limit(1)
    elif key == ord('h'):
        print("Homing")
        motion.homing()
    elif key == ord('f'):
        x, y = motion.get_position()
        save_calibration(x, y)
    elif key == ord('1'):
        print("Detect")
        vein = camera.detect_vein_points(model, img)
        print("Finished Detect")
    elif key == ord('2'):
        print("Plot first dot")
        target = map_vein_to_motion(vein, index=0)
        motion.move_to(target.x_mm, target.y_mm)
        servo.sweep_until_limit(direction=1)
        time.sleep(0.1)
        servo.set_angle(0)
        print("Finished Plot")
    elif key == ord('3'):
        print("Plot second dot")
        target = map_vein_to_motion(vein, index=1)
        motion.move_to(target.x_mm, target.y_mm)
        servo.sweep_until_limit(direction=1)
        time.sleep(0.1)
        servo.set_angle(0)
        print("Finished Plot")
    elif key == ord('z'):
        print("Quit")
        break


    if dx != 0 or dy != 0:
        motion.move_dir(dx, dy)

servo.cleanup()
cv2.destroyAllWindows()
