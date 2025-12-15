from src.stepper_motor import Motion2D
import cv2
import numpy as np

motion = Motion2D()

cv2.namedWindow("control", cv2.WINDOW_NORMAL)
cv2.imshow("control", np.zeros((100, 300), dtype=np.uint8))

print("Click the window. WASD to move, Q to quit")

while True:
    key = cv2.waitKey(50) & 0xFF

    dx, dy = 0, 0

    if key == ord('w'):
        dy = 1
    elif key == ord('s'):
        dy = -1
    elif key == ord('a'):
        dx = -1
    elif key == ord('d'):
        dx = 1
    elif key == ord('q'):
        print("Quit")
        break

    if dx != 0 or dy != 0:
        motion.dir(dx, dy)

cv2.destroyAllWindows()
