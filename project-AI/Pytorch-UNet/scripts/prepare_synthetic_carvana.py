import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_sample(size: int, rng: random.Random):
    bg = Image.new("RGB", (size, size), (rng.randint(20, 50), rng.randint(20, 50), rng.randint(35, 70)))
    mask = Image.new("L", (size, size), 0)
    draw_bg = ImageDraw.Draw(bg)
    draw_mask = ImageDraw.Draw(mask)

    for _ in range(18):
        x0 = rng.randint(0, size - 8)
        y0 = rng.randint(0, size - 8)
        x1 = min(size, x0 + rng.randint(8, 40))
        y1 = min(size, y0 + rng.randint(8, 40))
        color = tuple(rng.randint(30, 120) for _ in range(3))
        draw_bg.rectangle([x0, y0, x1, y1], fill=color)

    cx, cy = rng.randint(size // 4, size * 3 // 4), rng.randint(size // 4, size * 3 // 4)
    rx, ry = rng.randint(size // 7, size // 3), rng.randint(size // 8, size // 3)
    box = [cx - rx, cy - ry, cx + rx, cy + ry]
    fg = (rng.randint(150, 235), rng.randint(80, 180), rng.randint(80, 180))
    draw_bg.ellipse(box, fill=fg)
    draw_mask.ellipse(box, fill=255)

    for _ in range(rng.randint(1, 3)):
        pts = [(rng.randint(0, size), rng.randint(0, size)) for _ in range(3)]
        draw_bg.polygon(pts, fill=tuple(rng.randint(90, 220) for _ in range(3)))
        draw_mask.polygon(pts, fill=255)

    arr = np.asarray(bg).astype(np.int16)
    noise = np.random.default_rng(rng.randint(0, 10_000_000)).normal(0, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    bg = Image.fromarray(arr, "RGB").filter(ImageFilter.SMOOTH_MORE)
    return bg, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-dir", type=Path, default=Path("data/imgs"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/masks"))
    args = parser.parse_args()

    args.img_dir.mkdir(parents=True, exist_ok=True)
    args.mask_dir.mkdir(parents=True, exist_ok=True)
    for p in list(args.img_dir.glob("synthetic_*")) + list(args.mask_dir.glob("synthetic_*")):
        p.unlink()

    rng = random.Random(args.seed)
    for i in range(args.count):
        sample_id = f"synthetic_{i:04d}"
        image, mask = make_sample(args.size, rng)
        image.save(args.img_dir / f"{sample_id}.jpg", quality=95)
        mask.save(args.mask_dir / f"{sample_id}_mask.gif")

    print(f"generated {args.count} samples at {args.img_dir} and {args.mask_dir}")


if __name__ == "__main__":
    main()
