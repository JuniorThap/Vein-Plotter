import torch
import segmentation_models_pytorch as smp
from src.vein_selection import build_model


device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model("pretrained_unet_vein.pth")
dummy = torch.randn(1, 1, 480, 640).to(device)

torch.onnx.export(
    model,
    dummy,
    "vein_unet.onnx",
    input_names=["input"],
    output_names=["mask"],
    opset_version=11,
)
