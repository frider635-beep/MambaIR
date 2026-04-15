import os
import cv2
import torch
import numpy as np

from basicsr.archs import build_network
from basicsr.utils import img2tensor, tensor2img

MODEL_PATH = "experiments/pretrained_models/model.pth"
INPUT_PATH = "input.png"
OUTPUT_DIR = "results"
OUTPUT_PATH = "output.png"

DEVICE = torch.device("cpu")


def build_model():
    opt = {
        "type": "MambaIRv2",
        "img_size": 64,
        "in_chans": 3,
        "embed_dim": 48,
        "depths": [6, 6, 6, 6],
        "mlp_ratio": 2,
        "upscale": 1,
        "img_range": 1.0,
        "resi_connection": "1conv",
    }
    return build_network(opt)


def load_model(model):
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if "params" in checkpoint:
        state_dict = checkpoint["params"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(DEVICE)
    return model


def infer(model):
    img = cv2.imread(INPUT_PATH, cv2.IMREAD_COLOR)
    img = img.astype(np.float32) / 255.0

    img_tensor = img2tensor(img, bgr2rgb=True, float32=True)
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)

    output_img = tensor2img(output, rgb2bgr=True)
    return output_img


def main():
    print(" Loading model...")
    model = build_model()
    model = load_model(model)

    print(" Running inference...")
    result = infer(model)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_PATH)

    cv2.imwrite(save_path, result)
    print(f" Saved to {save_path}")


if __name__ == "__main__":
    main()