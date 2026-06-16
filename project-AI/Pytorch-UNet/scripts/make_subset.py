import argparse
import shutil
from pathlib import Path


def clear_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def link_or_copy(src: Path, dst: Path):
    try:
        if dst.exists():
            dst.unlink()
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-img-dir", type=Path, default=Path("data/carvana_real/imgs"))
    parser.add_argument("--src-mask-dir", type=Path, default=Path("data/carvana_real/masks"))
    parser.add_argument("--dst-img-dir", type=Path, required=True)
    parser.add_argument("--dst-mask-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args()

    clear_dir(args.dst_img_dir)
    clear_dir(args.dst_mask_dir)

    images = sorted([p for p in args.src_img_dir.iterdir() if p.is_file()])
    copied = 0
    for img in images:
        masks = list(args.src_mask_dir.glob(f"{img.stem}_mask.*"))
        if len(masks) != 1:
            continue
        link_or_copy(img, args.dst_img_dir / img.name)
        link_or_copy(masks[0], args.dst_mask_dir / masks[0].name)
        copied += 1
        if copied >= args.count:
            break

    print(f"subset_pairs={copied} dst_img_dir={args.dst_img_dir} dst_mask_dir={args.dst_mask_dir}")


if __name__ == "__main__":
    main()
