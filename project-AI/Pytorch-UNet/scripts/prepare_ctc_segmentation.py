import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def frame_id(path: Path) -> str:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse frame number from {path}")
    return digits


def save_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("L").save(dst)


def save_binary_mask(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(Image.open(src))
    Image.fromarray((arr > 0).astype(np.uint8)).save(dst)


def write_pair(
    image_src: Path,
    mask_src: Path,
    out_root: Path,
    sample_id: str,
    split: str,
    rows: list[dict[str, str]],
) -> None:
    image_dst = out_root / split / "imgs" / f"{sample_id}.png"
    mask_dst = out_root / split / "masks" / f"{sample_id}_mask.png"
    save_image(image_src, image_dst)
    save_binary_mask(mask_src, mask_dst)
    rows.append(
        {
            "split": split,
            "sample_id": sample_id,
            "image_src": str(image_src),
            "mask_src": str(mask_src),
            "image_dst": str(image_dst),
            "mask_dst": str(mask_dst),
        }
    )


def prepare_dataset(dataset_root: Path, out_root: Path, overwrite: bool = False) -> None:
    if overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    gt_frames: set[tuple[str, str]] = set()
    sequences = sorted(path.name for path in dataset_root.iterdir() if path.is_dir() and path.name in {"01", "02"})

    for seq in sequences:
        seg_dir = dataset_root / f"{seq}_GT" / "SEG"
        for seg_path in sorted(seg_dir.glob("man_seg*.tif")):
            fid = frame_id(seg_path)
            gt_frames.add((seq, fid))
            image_src = dataset_root / seq / f"t{fid}.tif"
            sample_id = f"{seq}_t{fid}"
            write_pair(image_src, seg_path, out_root, sample_id, "gt_seg", rows)
            write_pair(image_src, seg_path, out_root, sample_id, f"gt_seg_{seq}", rows)

    for seq in sequences:
        err_dir = dataset_root / f"{seq}_ERR_SEG"
        for mask_path in sorted(err_dir.glob("mask*.tif")):
            fid = frame_id(mask_path)
            image_src = dataset_root / seq / f"t{fid}.tif"
            sample_id = f"{seq}_t{fid}"
            write_pair(image_src, mask_path, out_root, sample_id, "err_all", rows)
            if (seq, fid) not in gt_frames:
                write_pair(image_src, mask_path, out_root, sample_id, "err_minus_gt", rows)

    manifest = out_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "sample_id", "image_src", "mask_src", "image_dst", "mask_dst"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    for split, count in sorted(counts.items()):
        print(f"{split}: {count}")
    print(f"manifest: {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare_dataset(args.dataset_root, args.out_root, args.overwrite)


if __name__ == "__main__":
    main()
