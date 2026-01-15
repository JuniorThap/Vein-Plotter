import glob
import os
import time
import pandas as pd
import cv2
import matplotlib.pyplot as plt

base_dir = "experiment2"
plot_files = sorted(glob.glob(os.path.join(base_dir, "*_plotted.png")))

results = []

def click_points(img, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

    points = plt.ginput(n=2, timeout=0)
    plt.close(fig)

    return [(int(x), int(y)) for x, y in points]

for plot_path in plot_files:
    visible_path = plot_path.replace("_plotted.png", "_visible.png")
    if not os.path.exists(visible_path):
        continue

    base_name = os.path.basename(plot_path).replace("_plotted.png", "")
    name = f"{base_dir}/{base_name}"

    plot_img = cv2.cvtColor(cv2.imread(plot_path), cv2.COLOR_BGR2RGB)
    visible_img = cv2.cvtColor(cv2.imread(visible_path), cv2.COLOR_BGR2RGB)

    print(f"\n{name}")

    visible_points = click_points(visible_img, "Click VISIBLE points (ENTER to finish)")
    plot_points = click_points(plot_img, "Click PLOT points (ENTER to finish)")

    results.append({
        "Name": name,
        "Plots": plot_points,
        "Visibles": visible_points,
    })

# ---- Save CSV ----
df = pd.DataFrame(results)
df["Plots"] = df["Plots"].apply(str)
df["Visibles"] = df["Visibles"].apply(str)

df.to_csv("experiment2_points.csv", index=False)

print("\nSaved to experiment2_points.csv")
