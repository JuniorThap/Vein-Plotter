import torch
import cv2
from skimage.morphology import skeletonize
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.export import load
try:
    import segmentation_models_pytorch as smp
except:
    pass

# region Model

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_model(model_path, auto=True, program=False):
    if not program:
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            activation="sigmoid"
        )
        if auto:
            model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    else:
        model = load(model_path)
        loaded_model = model.module()
        return loaded_model
# endregion


# region Data

def pad_to_divisible(img, div=32):
    h, w = img.shape[:2]
    pad_h = (div - h % div) % div
    pad_w = (div - w % div) % div

    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    return img, pad_h, pad_w

def preparedata(img):
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img, pad_h, pad_w = pad_to_divisible(img)
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float() / 255.0
    return img, (pad_h, pad_w)

def depad(img, pad_h, pad_w):
    if pad_h > 0:
        img = img[:-pad_h, :]
    if pad_w > 0:
        img = img[:, :-pad_w]
    return img

# endregion


# region Veins Selection

    # region Extract Veins

def get_skeleton(mask):
    # mask: binary (0/255)
    skel = skeletonize(mask > 0)
    return skel.astype(np.uint8)

def get_neighbors(y, x, skel):
    H, W = skel.shape
    neighbors = []
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                neighbors.append((ny, nx))
    return neighbors


def classify_skeleton_points(skel):
    endpoints = []
    junctions = []

    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        neighbors = get_neighbors(y, x, skel)
        n = len(neighbors)
        if n == 1:
            endpoints.append((y, x))
        elif n >= 3:
            for neighbor in neighbors:
                junctions.append(neighbor)

    return endpoints, junctions

def extract_segments(skel):
    visited = set()
    segments = []

    endpoints, junctions = classify_skeleton_points(skel)
    stop_points = set(endpoints + junctions)

    def walk(start):
        segment = [start]
        prev = None
        curr = start

        while True:
            visited.add(curr)
            neighbors = get_neighbors(curr[0], curr[1], skel)
            neighbors = [p for p in neighbors if p != prev]

            nxt = neighbors[0]
            if nxt in stop_points and nxt != start:
                segment.append(nxt)
                break

            segment.append(nxt)
            prev, curr = curr, nxt

        return segment

    for p in stop_points:
        if p not in visited:
            seg = walk(p)
            if len(seg) > 5:   # ignore tiny noise
                segments.append(seg)

    return segments

def fit_line_pca(points):
    pts = np.array([(x,y) for y,x in points])  # (N,2)

    mean = pts.mean(axis=0)
    centered = pts - mean

    _, S, Vt = np.linalg.svd(centered)
    direction = Vt[0]

    # perpendicular distance error
    proj = centered @ direction
    recon = np.outer(proj, direction)
    error = np.mean(np.linalg.norm(centered - recon, axis=1))

    length = proj.max() - proj.min()

    return {
        "center": mean,
        "direction": direction,
        "length": length,
        "error": error,
        "points": pts
    }

import numpy as np

def filter_points_by_slope(line, min_angle=15):
    """
    line: output dict from fit_line_pca
    min_angle: max allowed angle difference in degrees
    """
    pts = line["points"]              # (N,2) as (x,y)
    direction = line["direction"]     # PCA direction (unit-ish)

    direction = direction / np.linalg.norm(direction)

    keep = [True]   # first point always kept

    for i in range(1, len(pts)):
        v = pts[i] - pts[i-1]
        norm = np.linalg.norm(v)

        if norm < 1e-6:
            keep.append(False)
            continue

        v = v / norm

        # angle between local direction and PCA direction
        cos_angle = np.clip(np.abs(np.dot(v, direction)), 0, 1)
        angle = np.degrees(np.arccos(cos_angle))

        keep.append(angle <= min_angle)

    filtered_pts = pts[np.array(keep)]

    return filtered_pts

def extract_straight_vein_segments(mask, min_length=40):
    skel = get_skeleton(mask)
    segments = extract_segments(skel)

    lines = []
    for seg in segments:
        line = fit_line_pca(seg)

        if line["length"] <= min_length:   # minimum usable length
            continue

        filtered_pts = filter_points_by_slope(line, 30)

        if len(filtered_pts) < 5:
            continue

        filtered_seg = [(int(y), int(x)) for x, y in filtered_pts]
        refined_line = fit_line_pca(filtered_seg)

        # final length check (optional but recommended)
        if refined_line["length"] > min_length:
            lines.append(refined_line)

    return lines

    # endregion

    # region Merge Line

def angle_between_dirs(d1, d2):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)

    cos = np.clip(abs(np.dot(d1, d2)), 0, 1)
    return np.degrees(np.arccos(cos))


def endpoint_distance(line1, line2):
    p1 = line1["points"]
    p2 = line2["points"]

    ends1 = [p1[0], p1[-1]]
    ends2 = [p2[0], p2[-1]]

    return min(
        np.linalg.norm(e1 - e2)
        for e1 in ends1
        for e2 in ends2
    )

def projection_overlap(line1, line2, min_overlap=20):
    d = line1["direction"]
    p1 = (line1["points"] - line1["center"]) @ d
    p2 = (line2["points"] - line1["center"]) @ d

    overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
    return overlap > min_overlap

def mean_line_distance(line1, line2):
    d = line1["direction"]
    c = line1["center"]

    pts = line2["points"] - c
    perp = pts - np.outer(pts @ d, d)
    return np.mean(np.linalg.norm(perp, axis=1))


def merge_lines(lines, max_angle=15, max_dist=15):
    merged = []
    used = [False] * len(lines)

    for i, li in enumerate(lines):
        if used[i]:
            continue

        pts = li["points"].copy()
        used[i] = True

        for j, lj in enumerate(lines):
            if used[j] or i == j:
                continue

            if angle_between_dirs(li["direction"], lj["direction"]) > max_angle:
                continue

            temp_line = fit_line_pca([(y,x) for x,y in pts])
            if endpoint_distance(temp_line, lj) > max_dist:
                continue

            if not projection_overlap(temp_line, lj):
                continue

            if mean_line_distance(temp_line, lj) > 8:
                continue

            pts = np.vstack([pts, lj["points"]])

            if fit_line_pca(pts)["error"] > 3:
                continue

            used[j] = True

        merged.append(fit_line_pca([(y,x) for x,y in pts]))

    return merged

    # endregion

    # region Keep Straight Part

def keep_straightest_part(line, min_len=40, window_frac=0.6):
    pts = line["points"]
    if len(pts) < 10:
        return None

    # project points onto PCA direction
    d = line["direction"]
    c = line["center"]
    proj = (pts - c) @ d

    order = np.argsort(proj)
    pts_sorted = pts[order]

    n = len(pts_sorted)
    w = max(int(window_frac * n), 5)

    best = None
    best_err = np.inf

    for i in range(n - w + 1):
        seg = pts_sorted[i:i+w]
        test = fit_line_pca([(y, x) for x, y in seg])

        if test["length"] < min_len:
            continue

        if test["error"] < best_err:
            best_err = test["error"]
            best = test

    return best

    # endregion

    # region Select Top Veins
def angle_to_vertical(direction):
    d = direction / np.linalg.norm(direction)
    vertical = np.array([0.0, 1.0])
    cos = np.clip(abs(np.dot(d, vertical)), 0, 1)
    return np.degrees(np.arccos(cos))

import numpy as np

def min_distance_from_center_to_contour(points, hand_contour):
    pts = points.astype(np.float32)
    center = np.mean(pts, axis=0)
    contour = np.array(hand_contour).reshape(-1, 2).astype(np.float32)
    dists = np.linalg.norm(contour - center, axis=1)
    return float(np.min(dists))


def extract_hand_contour(gray):
    # Otsu threshold to separate hand from background
    _, hand = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Ensure hand is white
    if np.mean(gray[hand == 255]) < np.mean(gray[hand == 0]):
        hand = cv2.bitwise_not(hand)

    # Fill holes
    kernel = np.ones((7,7), np.uint8)
    hand = cv2.morphologyEx(hand, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered = []
    for c in contours:
        if cv2.contourArea(c) >= 20000:
            filtered.append(c)

    return filtered

def filter_near_hand_edge(lines, hand_contour, margin=15):
    filtered = []
    for line in lines:
        edge_dist = min_distance_from_center_to_contour(line["points"], hand_contour)
        if edge_dist < margin:
            continue
        filtered.append(line)
    return filtered

def score_vein(
    line,
    w_len=0.7,
    w_err=0, # 0.6
    w_ang=1,
):
    """
    Higher score = better vein
    """

    length = line["length"]
    error = line["error"] + 1e-6
    angle = angle_to_vertical(line["direction"])

    # normalize components
    len_score = length
    err_score = 1.0 / error
    ang_score = np.cos(np.radians(angle))   # 1 = vertical

    score = (
        w_len * len_score +
        w_err * err_score +
        w_ang * ang_score
    )

    return score

def select_top_vein(lines, k=3):
    if lines is None:
        return None
    if len(lines) == 0:
        return None

    scored = []
    for line in lines:
        s = score_vein(line)
        scored.append((s, line))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = []
    for score, line in scored[:k]:
        line = line.copy()
        line["score"] = score
        top.append(line)

    return top

    # endregion

    # region Display

import cv2
import numpy as np

def plot_vein(img, lines):
    # Make a copy so we don't modify original
    out_img = img.copy()

    # Ensure image is BGR (for colored drawing)
    if len(out_img.shape) == 2:
        out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)

    if lines is None:
        return out_img

    for idx, l in enumerate(lines, start=1):
        pts = l["points"]

        p0 = tuple(map(int, pts[0]))
        p1 = tuple(map(int, pts[-1]))

        # Draw line
        cv2.line( out_img, p0, p1, color=(0, 255, 0), thickness=2)

        # Midpoint for label
        mx = int((p0[0] + p1[0]) / 2)
        my = int((p0[1] + p1[1]) / 2)

        # Draw label background
        cv2.rectangle(out_img, (mx - 10, my - 10), (mx + 10, my + 10), (0, 0, 0), -1)

        # Draw index number
        cv2.putText(out_img, str(idx), (mx - 6, my + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return out_img

def show_segments(gt, img, lines, save_path=None):
    plt.figure(figsize=(12,6))

    if gt is not None:
        plt.subplot(1, 2, 1)
        plt.imshow(gt, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")
    
    if img is not None:
        plt.subplot(1, 2, 2)
        plt.imshow(img, cmap="gray")
        plt.title("Candidate straight vein segments")
        plt.axis("off")

        if lines is not None:
            for idx, l in enumerate(lines, start=1):
                pts = l["points"]
                p0 = pts[0]
                p1 = pts[-1]

                plt.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=2)

                mx = (p0[0] + p1[0]) / 2
                my = (p0[1] + p1[1]) / 2

                plt.text(
                    mx, my, f"{idx}",
                    color="yellow",
                    fontsize=12,
                    weight="bold",
                    ha="center", va="center",
                    bbox=dict(facecolor="black", alpha=0.5, pad=2)
                )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()

    # endregion

    # region Evaluate

def line_direction(p0, p1):
    d = np.array(p1) - np.array(p0)
    return d / (np.linalg.norm(d) + 1e-6)

def angle_between_lines(p0, p1, q0, q1):
    d1 = line_direction(p0, p1)
    d2 = line_direction(q0, q1)
    cos = np.clip(abs(np.dot(d1, d2)), 0, 1)
    return np.degrees(np.arccos(cos))

def segments_intersect(p1, p2, p3, p4, tol=5):
    def dist_point_to_line(pt, a, b):
        pt, a, b = map(np.array, (pt, a, b))
        d = b - a
        t = np.dot(pt - a, d) / (np.dot(d, d) + 1e-6)
        proj = a + np.clip(t, 0, 1) * d
        return np.linalg.norm(pt - proj)

    # check both directions
    return (
        dist_point_to_line(p1, p3, p4) < tol or
        dist_point_to_line(p2, p3, p4) < tol or
        dist_point_to_line(p3, p1, p2) < tol or
        dist_point_to_line(p4, p1, p2) < tol
    )

def evaluate_prediction(
    pred_line,        # [(x0,y0),(x1,y1)]
    gt_lines,         # list of GT lines, same format
    angle_thresh=15,
    dist_thresh=5
):
    """
    returns 1 if pred matches ANY GT line, else 0
    """

    p0, p1 = pred_line

    for gt in gt_lines:
        g0, g1 = gt

        # 1️⃣ slope similarity
        ang = angle_between_lines(p0, p1, g0, g1)
        if ang > angle_thresh:
            continue

        # 2️⃣ spatial intersection / proximity
        if segments_intersect(p0, p1, g0, g1, tol=dist_thresh):
            return 1   # MATCH FOUND

    return 0

    # endregion

# endregion


def pipeline(model, img, auto=True):
    input, pad_info = preparedata(img)
    pad_h, pad_w = pad_info
    if auto:
        input = input.to(device)
    
    with torch.no_grad():
        mask = model(input).detach().cpu().squeeze().numpy() > 0.5
    img = depad(img, pad_h, pad_w)
    mask = depad(mask, pad_h, pad_w)

    lines = extract_straight_vein_segments(mask, 50)
    lines = merge_lines(lines, max_angle=12, max_dist=5)

    trimmed = []
    for l in lines:
        core = keep_straightest_part(l, min_len=50)
        if core is not None:
            trimmed.append(core)
    
    hand_contours = extract_hand_contour(img)
    filtered = filter_near_hand_edge(trimmed, hand_contours[0], 50)
    top_lines =  select_top_vein(filtered, k=3)
    
    return plot_vein(img, filtered), top_lines
