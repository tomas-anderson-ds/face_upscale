import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from pathlib import Path
import onnxruntime as ort
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image, ImageChops
import time
import numpy as np
from model import UpscalerTransformer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Путь к входному изображению.",
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Путь к чекпоинту best.pth или last.pth.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="sr_output.png",
        help="Куда сохранить результат.",
    )

    # Если чекпоинт без config, можно задать вручную.
    parser.add_argument("--lr_size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--resize_type", type=int, default=Image.Resampling.BICUBIC)
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.weights, map_location="cpu")

    if isinstance(ckpt, dict):
        config = ckpt.get("config", {})
        state_dict = ckpt.get("model", ckpt)
    else:
        config = {}
        state_dict = ckpt

    model = UpscalerTransformer(
        dim=config.get("dim", args.dim),
        depth=config.get("depth", args.depth),
        num_heads=config.get("heads", args.heads),
        window_size=config.get("window", args.window),
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 128, 128)
    
    """torch.onnx.export(
        model,
        dummy_input,
        "last.onnx",
        export_params=True,
        opset_version=17,      # >= 16 для поддержки bicubic
        do_constant_folding=True,         # Оптимизация констант
        input_names=['input'],
        output_names=['output'],
    )"""    
       
    lr_size = config.get("lr_size", args.lr_size)

    image = Image.open(args.image).convert("RGB")
        
    image = image.resize((lr_size, lr_size), Image.Resampling.LANCZOS)
 
    image_256 = image.resize((lr_size * 2, lr_size * 2), args.resize_type)

    x = TF.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                sr = model(x)
        else:
            sr = model(x)
                 
    sr = torch.clamp(sr[0].float().cpu(), 0.0, 1.0)
    img_np = sr.permute(1, 2, 0).numpy()
    img_np = (img_np * 255.0).astype(np.uint8)
    if img_np.shape[2] == 1:
        img_np = img_np.squeeze(axis=2)
           
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = Image.new(image_256.mode, (512, 256))
    
    result.paste(image_256, (0, 0))      
    result.paste(Image.fromarray(img_np), (256, 0))
    result.save(out_path)
    
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()