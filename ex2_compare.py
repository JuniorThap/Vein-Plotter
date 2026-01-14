import pandas as pd
import cv2
import matplotlib.pyplot as plt
import ast

df = pd.read_csv("ex2_points.csv")

results = []

for idx, row in df.iterrows():
    name = row["Name"]
    n_points = int(row["Plots"])

    plot_img = cv2.cvtColor(
        cv2.imread(f"{name}_plotted.png"), cv2.COLOR_BGR2RGB
    )
    visible_img = cv2.cvtColor(
        cv2.imread(f"{name}_visible.png"), cv2.COLOR_BGR2RGB
    )

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # ---- Plot image (with numbers) ----
    axs[0].imshow(plot_img)
    axs[0].set_title("Plot (dot order)")
    axs[0].axis("off")

    h, w, _ = plot_img.shape
    for i in range(n_points):
        axs[0].text(
            10, 30 + i * 25,
            f"{i+1}",
            color="red",
            fontsize=14,
            weight="bold"
        )

    # ---- Visible image (click here) ----
    axs[1].imshow(visible_img)
    axs[1].set_title("Click points (press ENTER when done)")
    axs[1].axis("off")

    plt.tight_layout()

    print(f"\n{name}: Click {n_points} points, then press ENTER")

    # ginput waits for clicks, ENTER to finish
    clicked_points = plt.ginput(n=n_points, timeout=0)

    plt.close(fig)

    # Convert to int tuples
    clicked_points = [(int(x), int(y)) for x, y in clicked_points]

    results.append({
        "Name": name,
        "Points": clicked_points
    })

# ---- Save to CSV ----
out_df = pd.DataFrame(results)
out_df["Points"] = out_df["Points"].apply(str)
out_df.to_csv("clicked_points.csv", index=False)

print("\nSaved to clicked_points.csv")
