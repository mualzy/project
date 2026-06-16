import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def image_files(path: Path):
    return sorted([p for p in path.iterdir() if p.is_file() and not p.name.startswith(".")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=Path, default=Path("data/imgs"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/masks"))
    parser.add_argument("--out-dir", type=Path, default=Path("data_summary"))
    parser.add_argument("--max-visual", type=int, default=12)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "samples").mkdir(exist_ok=True)
    (args.out_dir / "visual_checks").mkdir(exist_ok=True)

    imgs = image_files(args.img_dir)
    rows = []
    anomalies = []
    size_counter = Counter()
    mask_values = Counter()

    for img_path in imgs:
        stem = img_path.stem
        masks = list(args.mask_dir.glob(f"{stem}_mask.*")) or list(args.mask_dir.glob(f"{stem}.*"))
        if len(masks) != 1:
            anomalies.append(f"{stem}: mask_count={len(masks)}")
            continue
        mask_path = masks[0]
        try:
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)
            arr = np.asarray(mask)
        except Exception as exc:
            anomalies.append(f"{stem}: unreadable={exc}")
            continue
        if img.size != mask.size:
            anomalies.append(f"{stem}: image_size={img.size}, mask_size={mask.size}")
        uniq = np.unique(arr)
        for v in uniq.tolist():
            mask_values[str(v)] += 1
        size_counter[f"{img.size[0]}x{img.size[1]}"] += 1
        rows.append({
            "id": stem,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "width": img.size[0],
            "height": img.size[1],
            "mask_unique_values": "|".join(map(str, uniq.tolist())),
            "mask_foreground_ratio": float((arr > 0).mean()),
        })

    with (args.out_dir / "sample_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "image_path", "mask_path", "width", "height", "mask_unique_values", "mask_foreground_ratio"])
        writer.writeheader()
        writer.writerows(rows)

    for row in rows[: args.max_visual]:
        img = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")
        overlay = img.copy()
        red = Image.new("RGB", img.size, (255, 0, 0))
        overlay = Image.composite(red, overlay, mask.point(lambda p: 110 if p > 0 else 0))
        panel = Image.new("RGB", (img.width * 3, img.height + 24), "white")
        panel.paste(img, (0, 24))
        panel.paste(mask.convert("RGB"), (img.width, 24))
        panel.paste(overlay, (img.width * 2, 24))
        draw = ImageDraw.Draw(panel)
        draw.text((6, 6), "image", fill=(0, 0, 0))
        draw.text((img.width + 6, 6), "mask", fill=(0, 0, 0))
        draw.text((img.width * 2 + 6, 6), "overlay", fill=(0, 0, 0))
        panel.save(args.out_dir / "visual_checks" / f"{row['id']}_panel.png")
        if row == rows[0]:
            img.save(args.out_dir / "samples" / f"{row['id']}_image.png")
            mask.save(args.out_dir / "samples" / f"{row['id']}_mask.png")

    overview = [
        "# Dataset Overview",
        "",
        f"- image_count: {len(imgs)}",
        f"- paired_valid_count: {len(rows)}",
        f"- anomalies: {len(anomalies)}",
        f"- image_size_distribution: {dict(size_counter)}",
        f"- mask_value_frequency: {dict(mask_values)}",
        f"- sample_stats: data_summary/sample_stats.csv",
        f"- visual_checks: data_summary/visual_checks/",
        "",
        "## Anomalies",
    ]
    overview.extend([f"- {a}" for a in anomalies] or ["- none"])
    (args.out_dir / "dataset_overview.md").write_text("\n".join(overview) + "\n", encoding="utf-8")
    print(f"valid_pairs={len(rows)} anomalies={len(anomalies)} sizes={dict(size_counter)}")


if __name__ == "__main__":
    main()
