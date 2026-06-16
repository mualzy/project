import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predict import predict_img
from unet import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    image_paths = sorted(
        [p for p in args.img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )[: args.limit]
    if not image_paths:
        raise RuntimeError(f"No images found in {args.img_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=args.classes)
    state_dict = torch.load(args.model, map_location=device)
    state_dict.pop("mask_values", None)
    model.load_state_dict(state_dict)
    model.to(device=device)
    model.eval()

    images = [Image.open(path).convert("RGB") for path in image_paths]
    for img in images[: args.warmup]:
        _ = predict_img(model, img, device, scale_factor=args.scale)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    per_image = []
    for path, img in zip(image_paths, images):
        start = time.perf_counter()
        _ = predict_img(model, img, device, scale_factor=args.scale)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        per_image.append((path.name, elapsed))

    peak_mib = (
        round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
        if device.type == "cuda"
        else ""
    )
    total = sum(v for _, v in per_image)
    avg = total / len(per_image)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "device",
                "scale",
                "images",
                "warmup",
                "total_seconds",
                "avg_seconds_per_image",
                "images_per_second",
                "peak_memory_mib",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": str(args.model),
                "device": str(device),
                "scale": args.scale,
                "images": len(per_image),
                "warmup": min(args.warmup, len(images)),
                "total_seconds": round(total, 4),
                "avg_seconds_per_image": round(avg, 4),
                "images_per_second": round(1 / avg, 4),
                "peak_memory_mib": peak_mib,
            }
        )

    print(
        f"images={len(per_image)} avg_seconds_per_image={avg:.4f} "
        f"images_per_second={1 / avg:.4f} peak_memory_mib={peak_mib}"
    )


if __name__ == "__main__":
    main()
