import pandas as pd
import ast
import numpy as np
from src.mapping import pixel_to_mm, IMG_H, IMG_W, FIELD_X_MM, FIELD_Y_MM

# ---------------- LOAD DATA ----------------
df = pd.read_csv("experiment2_points.csv")

def parse_points(s):
    if pd.isna(s) or s.strip() == "[]":
        return []
    return ast.literal_eval(s)

df["Plots"] = df["Plots"].apply(parse_points)
df["Visibles"] = df["Visibles"].apply(parse_points)

# ---------------- CONTAINERS ----------------
start_px, end_px = [], []
start_mm, end_mm = [], []
slope_deg = []

# Field diagonal (for percent normalization)
FIELD_DIAG_MM = np.sqrt(FIELD_X_MM**2 + FIELD_Y_MM**2)

# ---------------- MAIN LOOP ----------------
for _, row in df.iterrows():
    plot = row["Plots"]
    vis = row["Visibles"]

    if len(plot) != 2 or len(vis) != 2:
        continue

    # ---------- START POINT ----------
    dx_px = plot[0][0] - vis[0][0]
    dy_px = plot[0][1] - vis[0][1]

    err_px = np.sqrt(dx_px**2 + dy_px**2)

    dx_mm = pixel_to_mm(abs(dx_px), IMG_W, FIELD_X_MM)
    dy_mm = pixel_to_mm(abs(dy_px), IMG_H, FIELD_Y_MM)
    err_mm = np.sqrt(dx_mm**2 + dy_mm**2)

    start_px.append(err_px)
    start_mm.append(err_mm)

    # ---------- END POINT ----------
    dx_px = plot[1][0] - vis[1][0]
    dy_px = plot[1][1] - vis[1][1]

    err_px = np.sqrt(dx_px**2 + dy_px**2)

    dx_mm = pixel_to_mm(abs(dx_px), IMG_W, FIELD_X_MM)
    dy_mm = pixel_to_mm(abs(dy_px), IMG_H, FIELD_Y_MM)
    err_mm = np.sqrt(dx_mm**2 + dy_mm**2)

    end_px.append(err_px)
    end_mm.append(err_mm)

    # ---------- SLOPE / DIRECTION ----------
    dx_gt = plot[1][0] - plot[0][0]
    dy_gt = plot[1][1] - plot[0][1]

    dx_pr = vis[1][0] - vis[0][0]
    dy_pr = vis[1][1] - vis[0][1]

    theta_gt = np.arctan2(dy_gt, dx_gt)
    theta_pr = np.arctan2(dy_pr, dx_pr)

    dtheta = np.abs(theta_gt - theta_pr)
    dtheta = np.minimum(dtheta, np.pi - dtheta)  # wrap to [0, pi]

    slope_deg.append(np.degrees(dtheta))

# ---------------- TO ARRAYS ----------------
start_px = np.array(start_px)
end_px = np.array(end_px)
start_mm = np.array(start_mm)
end_mm = np.array(end_mm)
slope_deg = np.array(slope_deg)

all_px = np.concatenate([start_px, end_px])
all_mm = np.concatenate([start_mm, end_mm])
all_pct = (all_mm / FIELD_DIAG_MM) * 100

# ---------------- METRICS ----------------
def summarize(arr):
    return arr.mean(), np.median(arr), arr.std(ddof=1)

metrics = {
    "START px": summarize(start_px),
    "START mm": summarize(start_mm),
    "START %": summarize((start_mm / FIELD_DIAG_MM) * 100),

    "END px": summarize(end_px),
    "END mm": summarize(end_mm),
    "END %": summarize((end_mm / FIELD_DIAG_MM) * 100),

    "OVERALL px": summarize(all_px),
    "OVERALL mm": summarize(all_mm),
    "OVERALL %": summarize(all_pct),

    "SLOPE deg": summarize(slope_deg),
}

# ---------------- PRINT ----------------
print("\n=== ACCURACY & PRECISION ===\n")

for k, (mean, median, std) in metrics.items():
    print(f"{k:<12} | Mean: {mean:.3f} | Median: {median:.3f} | STD: {std:.3f}")

print(f"\nValid pairs: {len(start_px)}")
print(f"Total points: {len(all_px)}")
