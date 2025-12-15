from src.vein_selection import build_model, pipeline
from glob import glob
import matplotlib.pyplot as plt
import cv2
import os

paths = glob("experiment1_1\experiment\*.png")
print("Path Length:", len(paths))
print("Example:", paths[0])

model = build_model("pretrained_unet_vein.pth")

save_dir = "experiment1_1\outputs"
os.makedirs(save_dir, exist_ok=True)

for path in paths:
    if "_IR" in path:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        output, lines = pipeline(model, img)
        
        basename = os.path.basename(path)
        save_path = os.path.join(save_dir, basename)
        
        cv2.imwrite(save_path, output)
        print("Save output at:", save_path)

        if False:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(img)

            plt.subplot(1, 2, 2)
            plt.imshow(output)
            plt.show()