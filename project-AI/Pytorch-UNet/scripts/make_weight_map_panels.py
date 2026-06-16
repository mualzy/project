import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def normalize_weight(weight: np.ndarray) -> Image.Image:
    weight = weight.astype(np.float32, copy=False)
    lo = float(np.percentile(weight, 1))
    hi = float(np.percentile(weight, 99))
    if hi <= lo:
        hi = float(weight.max())
        lo = float(weight.min())
    if hi <= lo:
        scaled = np.zeros_like(weight, dtype=np.uint8)
    else:
        scaled = np.clip((weight - lo) / (hi - lo), 0, 1)
        scaled = (scaled * 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L")


def display_mask(img: Image.Image) -> Image.Image:
    if img.getextrema()[1] <= 1:
        return img.point(lambda p: p * 255)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--weight-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures/weight_maps"))
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--prefix", default="exp")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    weights = sorted(args.weight_dir.glob("*_weight.npy"))[: args.limit]
    for i, weight_path in enumerate(weights, 1):
        stem = weight_path.stem.removesuffix("_weight")
        img_path = next(args.img_dir.glob(f"{stem}.*"))
        mask_path = (list(args.mask_dir.glob(f"{stem}_mask.*")) or list(args.mask_dir.glob(f"{stem}.*")))[0]

        img = Image.open(img_path).convert("RGB")
        mask = display_mask(Image.open(mask_path).convert("L")).resize(img.size, resample=Image.NEAREST)
        weight = normalize_weight(np.load(weight_path)).resize(img.size, resample=Image.BICUBIC)

        panel = Image.new("RGB", (img.width * 3, img.height + 24), "white")
        panel.paste(img, (0, 24))
        panel.paste(mask.convert("RGB"), (img.width, 24))
        panel.paste(weight.convert("RGB"), (img.width * 2, 24))
        draw = ImageDraw.Draw(panel)
        draw.text((6, 6), "image", fill=(0, 0, 0))
        draw.text((img.width + 6, 6), "weak mask", fill=(0, 0, 0))
        draw.text((img.width * 2 + 6, 6), "border weight", fill=(0, 0, 0))
        panel.save(args.out_dir / f"{args.prefix}_weight_sample_{i:02d}.png")

    print(f"created {len(weights)} panels in {args.out_dir}")


if __name__ == "__main__":
    main()
