from src.stepper_motor import Motion2D
import cv2

motion = Motion2D()

print("WASD to move, Q to quit")

while True:
    key = cv2.waitKey(50) & 0xFF  # 50 ms polling

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
