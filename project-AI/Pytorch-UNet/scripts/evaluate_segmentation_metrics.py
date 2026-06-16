import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unet import build_unet
from utils.data_loading import BasicDataset
from utils.overlap_tile import overlap_tile_logits


def parse_mask_values(raw):
    if raw is None:
        return None
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def parse_angles(raw):
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def find_mask(mask_dir, image_stem):
    candidates = list(mask_dir.glob(f"{image_stem}_mask.*")) or list(mask_dir.glob(f"{image_stem}.*"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one mask for {image_stem}, found {len(candidates)}")
    return candidates[0]


def map_mask(mask_path, mask_values, scale):
    mask = Image.open(mask_path)
    return BasicDataset.preprocess(mask_values, mask, scale, is_mask=True)


def center_crop_np(arr, target_shape):
    target_h, target_w = target_shape
    h, w = arr.shape[-2:]
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return arr[..., top:top + target_h, left:left + target_w]


def predict_logits(model, image_tensor, device, amp, tta_angles, overlap_tile=False, tile_size=572):
    logits_sum = None
    with torch.inference_mode(), torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for angle in tta_angles:
            if angle:
                aug_img = TF.rotate(image_tensor, angle=angle)
            else:
                aug_img = image_tensor
            if overlap_tile:
                logits = overlap_tile_logits(model, aug_img, device=device, tile_size=tile_size, amp=amp)
            else:
                aug_img = aug_img.unsqueeze(0).to(
                    device=device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )
                logits = model(aug_img)
            if angle:
                logits = TF.rotate(logits, angle=-angle)
            logits_sum = logits if logits_sum is None else logits_sum + logits
    return logits_sum / len(tta_angles)


def binary_metrics(pred, truth):
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    intersection = np.logical_and(pred, truth).sum()
    pred_sum = pred.sum()
    truth_sum = truth.sum()
    union = np.logical_or(pred, truth).sum()
    dice = (2 * intersection) / max(pred_sum + truth_sum, 1)
    iou = intersection / max(union, 1)
    pixel_error = np.not_equal(pred, truth).mean()
    return dice, iou, pixel_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--architecture", choices=["milesial", "original"], default=None)
    parser.add_argument("--overlap-tile", action="store_true")
    parser.add_argument("--tile-size", type=int, default=572)
    parser.add_argument("--mask-values", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tta-angles", type=str, default="0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    state_dict = torch.load(args.model, map_location=device)
    state_mask_values = state_dict.pop("mask_values", None)
    checkpoint_architecture = state_dict.pop("architecture", None)
    mask_values = parse_mask_values(args.mask_values) or state_mask_values or [0, 255]
    architecture = args.architecture or checkpoint_architecture or "milesial"

    model = build_unet(architecture, n_channels=args.channels, n_classes=args.classes)
    model.load_state_dict(state_dict)
    model.to(device=device, memory_format=torch.channels_last)
    model.eval()

    tta_angles = parse_angles(args.tta_angles)
    if not tta_angles:
        tta_angles = [0.0]

    if args.pred_dir:
        args.pred_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([path for path in args.img_dir.iterdir() if path.is_file()])
    rows = []
    start = time.perf_counter()
    for image_path in image_paths:
        mask_path = find_mask(args.mask_dir, image_path.stem)
        image = Image.open(image_path)
        image_tensor = torch.as_tensor(
            BasicDataset.preprocess(mask_values, image, args.scale, is_mask=False).copy()
        ).float()
        truth = map_mask(mask_path, mask_values, args.scale)

        logits = predict_logits(
            model,
            image_tensor,
            device,
            args.amp,
            tta_angles,
            overlap_tile=args.overlap_tile or architecture == "original",
            tile_size=args.tile_size,
        )
        if args.classes == 1:
            pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        else:
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        if truth.shape != pred.shape:
            truth = center_crop_np(truth, pred.shape)

        dice, iou, pixel_error = binary_metrics(pred == 1, truth == 1)
        rows.append(
            {
                "image": image_path.name,
                "mask": mask_path.name,
                "dice": dice,
                "iou": iou,
                "pixel_error": pixel_error,
            }
        )

        if args.pred_dir:
            pred_img = Image.fromarray((pred * 255).astype(np.uint8))
            pred_img.save(args.pred_dir / f"{image_path.stem}.png")

    runtime = time.perf_counter() - start
    mean_dice = float(np.mean([row["dice"] for row in rows])) if rows else float("nan")
    mean_iou = float(np.mean([row["iou"] for row in rows])) if rows else float("nan")
    mean_pixel_error = float(np.mean([row["pixel_error"] for row in rows])) if rows else float("nan")

    with args.out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image",
            "mask",
            "dice",
            "iou",
            "pixel_error",
            "mean_dice",
            "mean_iou",
            "mean_pixel_error",
            "runtime_seconds",
            "tta_angles",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "mean_dice": mean_dice,
                    "mean_iou": mean_iou,
                    "mean_pixel_error": mean_pixel_error,
                    "runtime_seconds": runtime,
                    "tta_angles": ",".join(str(angle) for angle in tta_angles),
                }
            )

    print(
        "images={} mean_dice={:.6f} mean_iou={:.6f} mean_pixel_error={:.6f} runtime_seconds={:.4f}".format(
            len(rows), mean_dice, mean_iou, mean_pixel_error, runtime
        )
    )


if __name__ == "__main__":
    main()
