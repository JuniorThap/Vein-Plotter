from src.stepper_motor import Motion2D
from src.servo_motor import ServoWithLimit
from src.image_pipeline import Camera
from src.display_emer import UI
from src.experiment import Experiment
from src.stepper_calibration import save_calibration
from src.vein_selection import build_model
from src.mapping import map_vein_to_motion
import cv2
import numpy as np
import time
import pandas as pd


print("Click the window. WASD to move, QE for servo, Z to quit")

KEY_UP    = 82
KEY_DOWN  = 84
KEY_LEFT  = 81
KEY_RIGHT = 83
KEY_ENTER = 13        # sometimes 10 on Linux

motion = Motion2D()
servo = ServoWithLimit()
camera = Camera()
ui = UI()

ui.green_on()
model = build_model("", program=True)

experiment = False
save_dirs = ["img_log", "experiment2"]
log = Experiment(save_dirs[0], toggle_hand_side=False)
experiment2 = Experiment(save_dirs[1], toggle_hand_side=True, start_hand_index=1)
df = pd.read_csv("experiment2_points.csv")

# Main Control
while True:
    img, gray = camera.capture_image()
    cv2.imshow("Camera", img)
    key = cv2.waitKey(1) & 0xFFFFFFFF
    dx, dy = 0, 0

    log_object = experiment2 if experiment else log

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
    elif key == ord('c'):
        cv2.destroyWindow("Captured")  # safe even if it doesn't exist
        cv2.namedWindow("Captured", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Captured", img.shape[1], img.shape[0])
        cv2.imshow("Captured", img)

        start_file_name = log_object.get_start_filename()
        cv2.imwrite(start_file_name + "_visible.png", img)
        log_object.update_dir()

        cv2.waitKey(1)
    elif key == ord('i'):
        camera.ir_toggle()
    
    elif key == ord('h'):
        print("Set Home")
        motion.set_home()

    elif key == ord('g'):
        print("Homing")
        motion.go_home()
    
    elif key == ord('f'):
        x, y = motion.get_position()
        save_calibration(x, y)
        motion.set_offset(x, y)

    elif key == KEY_UP:
        motion.move_offset(0, -1)
    elif key == KEY_DOWN:
        motion.move_offset(0, 1)
    elif key == KEY_LEFT:
        motion.move_offset(-1, 0)
    elif key == KEY_RIGHT:
        motion.move_offset(1, 0)
    elif key == KEY_ENTER:
        motion.save_offset()

    elif key == ord('p'):
        experiment = not experiment
        print("Is experiment:", experiment)
    
    elif key == ord('1'):
        print("Detect")
        ui.yellow_on()
        vein, plotted = camera.detect_vein_points(model, gray)

        start_file_name = log_object.get_start_filename()

        cv2.imwrite(start_file_name + "_plotted.png", plotted)

        cv2.destroyWindow("Plotted")
        cv2.namedWindow("Plotted", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Plotted", plotted.shape[1], plotted.shape[0])
        cv2.imshow("Plotted", plotted)
        cv2.waitKey(1)

        print("Finished Detect")
        ui.yellow_off()
    elif key == ord('2'):
        print("Plot first dot")
        ui.yellow_on()
        target = map_vein_to_motion(vein, index=0)
        motion.move_to(target.x_mm, target.y_mm)
        servo.sweep_until_limit(direction=1)
        time.sleep(0.1)
        servo.set_angle(0)
        print("Finished Plot")
        ui.yellow_off()
    elif key == ord('3'):
        print("Plot second dot")
        ui.yellow_on()
        target = map_vein_to_motion(vein, index=1)
        motion.move_to(target.x_mm, target.y_mm)
        servo.sweep_until_limit(direction=1)
        time.sleep(0.1)
        servo.set_angle(0)
        print("Finished Plot")
        ui.yellow_off()

    elif ui.is_button_pressed() or key == ord('4'):
        start = time.perf_counter()
        print("Detect")
        ui.yellow_on()
        vein, plotted = camera.detect_vein_points(model, gray)

        start_file_name = log_object.get_start_filename()

        cv2.imwrite(start_file_name + "_plotted.png", plotted)

        cv2.destroyWindow("Plotted")
        cv2.namedWindow("Plotted", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Plotted", plotted.shape[1], plotted.shape[0])
        cv2.imshow("Plotted", plotted)
        cv2.waitKey(1)

        print("Finished Detect")
        ui.yellow_off()

        print("Plot first dot")
        ui.yellow_on()
        target = map_vein_to_motion(vein, index=0)
        motion.move_to(target.x_mm, target.y_mm)
        servo.sweep_until_limit(direction=1)
        time.sleep(0.1)
        servo.set_angle(0)
        print("Finished Plot")

        print("Plot second dot")
        ui.yellow_on()
        target = map_vein_to_motion(vein, index=1)
        motion.move_to(target.x_mm, target.y_mm)
        servo.sweep_until_limit(direction=1)
        time.sleep(0.1)
        servo.set_angle(0)
        print("Finished Plot")
        ui.yellow_off()

        print("Save INFO")
        duration = time.perf_counter() - start
        print("Duration:", duration, "s")
        new_row = pd.DataFrame({"Name": [start_file_name], "Plots": [vein.points_px], "Visibles":[None], "Time":[duration]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv("experiment2_points.csv", index=False)
        print("ALL DONE!")

    elif key == ord('z'):
        print("Quit")
        break

    

    if dx != 0 or dy != 0:
        motion.move_dir(dx, dy)

servo.cleanup()
cv2.destroyAllWindows()
