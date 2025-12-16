import torch
from torch.export import export, save, load
from src.vein_selection import build_model, pipeline
import cv2


model = build_model("pretrained_unet_vein.pth", auto=False)
model.eval()
sample_input = torch.randn(1, 1, 480, 640)


exported_program = export(model, (sample_input,))

save(exported_program, 'pretrained_unet_vein.pt2')

model = load('pretrained_unet_vein.pt2')
loaded_model = model.module()

img = cv2.imread(r"experiment1_1\high_contrast\person_001_L1_IR.png", cv2.IMREAD_GRAYSCALE)
out, lines = pipeline(loaded_model, img, auto=False)
cv2.imshow("Output", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

