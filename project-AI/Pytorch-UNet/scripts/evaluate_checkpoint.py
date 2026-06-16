import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import evaluate
from unet import build_unet
from utils.data_loading import BasicDataset, CarvanaDataset


def parse_mask_values(raw):
    if raw is None:
        return None
    values = []
    for value in raw.split(","):
        value = value.strip()
        if value:
            values.append(int(value))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--validation", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--architecture", choices=["milesial", "original"], default=None)
    parser.add_argument("--mask-values", type=str, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_false", dest="persistent_workers", default=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    mask_values = parse_mask_values(args.mask_values)

    try:
        dataset = CarvanaDataset(args.img_dir, args.mask_dir, args.scale, mask_values=mask_values)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(args.img_dir, args.mask_dir, args.scale, mask_values=mask_values)

    n_val = int(len(dataset) * (args.validation / 100))
    n_train = len(dataset) - n_val
    _, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    drop_last = len(val_set) >= args.batch_size
    loader_args = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
    )
    if args.num_workers > 0:
        loader_args["persistent_workers"] = args.persistent_workers
        loader_args["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(val_set, **loader_args)

    state_dict = torch.load(args.model, map_location=device)
    state_dict.pop("mask_values", None)
    checkpoint_architecture = state_dict.pop("architecture", None)
    architecture = args.architecture or checkpoint_architecture or "milesial"
    model = build_unet(architecture, n_channels=args.channels, n_classes=args.classes)
    model.load_state_dict(state_dict)
    model.to(device=device, memory_format=torch.channels_last)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    dice = evaluate(model, loader, device, args.amp)
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime = time.perf_counter() - start
    peak_mib = (
        round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
        if device.type == "cuda"
        else ""
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "dataset_size",
                "validation_size",
                "evaluated_batches",
                "batch_size",
                "channels",
                "scale",
                "amp",
                "num_workers",
                "device",
                "dice",
                "runtime_seconds",
                "peak_memory_mib",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": str(args.model),
                "dataset_size": len(dataset),
                "validation_size": n_val,
                "evaluated_batches": len(loader),
                "batch_size": args.batch_size,
                "channels": args.channels,
                "scale": args.scale,
                "amp": args.amp,
                "num_workers": args.num_workers,
                "device": str(device),
                "dice": float(dice),
                "runtime_seconds": round(runtime, 4),
                "peak_memory_mib": peak_mib,
            }
        )

    print(f"dice={float(dice):.6f} runtime_seconds={runtime:.4f} peak_memory_mib={peak_mib}")


if __name__ == "__main__":
    main()
