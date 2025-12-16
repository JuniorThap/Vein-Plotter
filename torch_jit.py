import torch
import torch
from src.vein_selection import build_model, pipeline
import cv2


# model = build_model("pretrained_unet_vein.pth", auto=False, program=False)
# model.eval()

# sample_input = torch.randn(1, 1, 480, 640)
# traced = torch.jit.trace(model, sample_input)
# traced.save("vein_unet_ts.pt")

# model = torch.jit.load("vein_unet_ts.pt", map_location="cpu")
model = build_model("", program=True)

img = cv2.imread(r"experiment1_1\high_contrast\person_001_L1_IR.png", cv2.IMREAD_GRAYSCALE)
out, lines = pipeline(model, img, auto=False)
cv2.imshow("Output", out)
cv2.waitKey(0)
cv2.destroyAllWindows()