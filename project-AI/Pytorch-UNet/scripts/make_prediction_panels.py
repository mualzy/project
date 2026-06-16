import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def display_mask(img):
    if img.getextrema()[1] <= 1:
        return img.point(lambda p: p * 255)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=Path, default=Path("data/imgs"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/masks"))
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures/prediction_samples"))
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--prefix", default="exp")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    preds = sorted(args.pred_dir.glob("*.png"))[: args.limit]
    for i, pred_path in enumerate(preds, 1):
        stem = pred_path.stem.replace("_OUT", "")
        img_path = next(args.img_dir.glob(f"{stem}.*"))
        mask_path = (list(args.mask_dir.glob(f"{stem}_mask.*")) or list(args.mask_dir.glob(f"{stem}.*")))[0]
        img = Image.open(img_path).convert("RGB")
        truth = display_mask(Image.open(mask_path).convert("L")).resize(img.size)
        pred = display_mask(Image.open(pred_path).convert("L")).resize(img.size)
        panel = Image.new("RGB", (img.width * 3, img.height + 24), "white")
        panel.paste(img, (0, 24))
        panel.paste(truth.convert("RGB"), (img.width, 24))
        panel.paste(pred.convert("RGB"), (img.width * 2, 24))
        draw = ImageDraw.Draw(panel)
        draw.text((6, 6), "image", fill=(0, 0, 0))
        draw.text((img.width + 6, 6), "ground truth", fill=(0, 0, 0))
        draw.text((img.width * 2 + 6, 6), "prediction", fill=(0, 0, 0))
        panel.save(args.out_dir / f"{args.prefix}_pred_sample_{i:02d}.png")
    print(f"created {len(preds)} panels in {args.out_dir}")


if __name__ == "__main__":
    main()
