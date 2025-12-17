from src.vein_selection import pipeline, build_model
import cv2

model = build_model("pretrained_unet_vein.pth")
input = cv2.imread("experiment1_1\experiment\person_002_R1_IR.png", cv2.IMREAD_GRAYSCALE)
out, lines = pipeline(model, input)