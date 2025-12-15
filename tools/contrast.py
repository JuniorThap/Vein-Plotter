from glob import glob
import cv2
import os

paths = glob("experiment1_1/experiment/*.png")
print("Path Length:", len(paths))
print("Example:", paths[0])

save_dir = "experiment1_1/high_contrast"
os.makedirs(save_dir, exist_ok=True)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

for path in paths:
    if "_IR" not in path:
        continue

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed:", path)
        continue

    # 🔥 High contrast
    output = clahe.apply(img)

    basename = os.path.basename(path)
    save_path = os.path.join(save_dir, basename)

    cv2.imwrite(save_path, output)
    print("Save output at:", save_path)
