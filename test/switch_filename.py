import os
import cv2
from glob import glob


# Get Target
experiment_dir = "experiment1_1\experiment"
target_ids = [3, 5, 7, 8]
ids = [f"{id:03}" for id in target_ids]
print("Target IDs:", ids)

format = os.path.join(experiment_dir, "person_{}_{}_{}.png")
for id in ids:
    os.rename(format.format(id, "L1", "IR"), format.format(id, "T1", "IR"))
    os.rename(format.format(id, "R1", "IR"), format.format(id, "L1", "IR"))
    os.rename(format.format(id, "T1", "IR"), format.format(id, "R1", "IR"))
    
    os.rename(format.format(id, "L1", "NoIR"), format.format(id, "T1", "NoIR"))
    os.rename(format.format(id, "R1", "NoIR"), format.format(id, "L1", "NoIR"))
    os.rename(format.format(id, "T1", "NoIR"), format.format(id, "R1", "NoIR"))
    print("Change ID:", id)