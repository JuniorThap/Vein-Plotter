import cv2
import numpy as np
import json

HSV_SAVE_FILE = "hsv_range.json"


def hsv_area_tuner(camera_id=0, width=640, height=480):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    roi_size = 10  # size of sampling square (px)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2

        # ROI box
        x1 = cx - roi_size // 2
        y1 = cy - roi_size // 2
        x2 = cx + roi_size // 2
        y2 = cy + roi_size // 2

        roi = frame[y1:y2, x1:x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Mean HSV of ROI
        mean_hsv = np.mean(hsv_roi.reshape(-1, 3), axis=0).astype(int)
        h_mean, s_mean, v_mean = mean_hsv

        # Auto range (tunable margins)
        h_margin = 10
        s_margin = 40
        v_margin = 40

        lower = np.array([
            max(h_mean - h_margin, 0),
            max(s_mean - s_margin, 0),
            max(v_mean - v_margin, 0),
        ])

        upper = np.array([
            min(h_mean + h_margin, 179),
            min(s_mean + s_margin, 255),
            min(v_mean + v_margin, 255),
        ])

        # Mask using auto HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        masked = cv2.bitwise_and(frame, frame, mask=mask)

        # Draw ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Color preview block (convert HSV → BGR)
        color_patch = np.zeros((100, 100, 3), dtype=np.uint8)
        color_patch[:] = cv2.cvtColor(
            np.uint8([[[h_mean, s_mean, v_mean]]]),
            cv2.COLOR_HSV2BGR
        )

        # Display
        vis = np.hstack([
            cv2.resize(frame, (320, 240)),
            cv2.cvtColor(cv2.resize(mask, (320, 240)), cv2.COLOR_GRAY2BGR),
            cv2.resize(color_patch, (320, 240)),
        ])

        cv2.imshow("HSV Area Tuner | Frame | Mask | Color", vis)

        print(f"Mean HSV: H={h_mean}, S={s_mean}, V={v_mean}", end="\r")

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            data = {
                "mean_hsv": [int(h_mean), int(s_mean), int(v_mean)],
                "lower": lower.tolist(),
                "upper": upper.tolist()
            }
            with open(HSV_SAVE_FILE, "w") as f:
                json.dump(data, f, indent=4)
            print("\nSaved HSV range:", data)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    hsv_area_tuner(camera_id=0)
