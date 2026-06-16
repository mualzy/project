import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


def resolve_path(path_text: str, manifest: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path
    candidate = manifest.parent / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(path_text)


def resize_label(label: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return label
    h, w = label.shape
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = Image.fromarray(label.astype(np.uint16)).resize((new_w, new_h), resample=Image.NEAREST)
    return np.asarray(resized, dtype=np.int32)


def resize_float(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(arr.astype(np.float32), mode="F").resize(size, resample=Image.BICUBIC),
        dtype=np.float32,
    )


def boundary_coords(mask: np.ndarray, max_points: int) -> np.ndarray:
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    boundary = mask & ~interior
    coords = np.argwhere(boundary)
    if len(coords) == 0:
        coords = np.argwhere(mask)
    if len(coords) > max_points:
        stride = math.ceil(len(coords) / max_points)
        coords = coords[::stride][:max_points]
    return coords.astype(np.float32)


def nearest_boundary_distance_sq(
    pixel_y: np.ndarray,
    pixel_x: np.ndarray,
    coords: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    out = np.empty(pixel_y.shape[0], dtype=np.float32)
    by = coords[:, 0]
    bx = coords[:, 1]
    for start in range(0, pixel_y.shape[0], chunk_size):
        end = min(start + chunk_size, pixel_y.shape[0])
        dy = pixel_y[start:end, None] - by[None, :]
        dx = pixel_x[start:end, None] - bx[None, :]
        out[start:end] = np.min(dy * dy + dx * dx, axis=1)
    return out


def make_weight_map(
    label: np.ndarray,
    w0: float,
    sigma: float,
    distance_scale: float,
    max_boundary_points: int,
    chunk_size: int,
    class_weight_bg: float,
    class_weight_fg: float,
) -> tuple[np.ndarray, dict[str, float]]:
    label = label.astype(np.int32, copy=False)
    h, w = label.shape
    small = resize_label(label, distance_scale)
    sh, sw = small.shape

    ids = [int(v) for v in np.unique(small) if int(v) != 0]
    border_small = np.zeros((sh, sw), dtype=np.float32)
    if len(ids) >= 2 and w0 > 0:
        yy, xx = np.indices((sh, sw), dtype=np.float32)
        pixel_y = yy.reshape(-1)
        pixel_x = xx.reshape(-1)
        d1 = np.full(pixel_y.shape, np.inf, dtype=np.float32)
        d2 = np.full(pixel_y.shape, np.inf, dtype=np.float32)

        for obj_id in ids:
            coords = boundary_coords(small == obj_id, max_points=max_boundary_points)
            if len(coords) == 0:
                continue
            dist_sq = nearest_boundary_distance_sq(pixel_y, pixel_x, coords, chunk_size=chunk_size)
            closer = dist_sq < d1
            d2[closer] = d1[closer]
            d1[closer] = dist_sq[closer]
            second = (~closer) & (dist_sq < d2)
            d2[second] = dist_sq[second]

        valid = np.isfinite(d1) & np.isfinite(d2)
        sigma_small = max(float(sigma) * float(distance_scale), 1e-6)
        dsum = np.sqrt(d1[valid]) + np.sqrt(d2[valid])
        border_flat = border_small.reshape(-1)
        border_flat[valid] = w0 * np.exp(-(dsum * dsum) / (2.0 * sigma_small * sigma_small))

    if border_small.shape != label.shape:
        border = resize_float(border_small, (w, h))
        border = np.maximum(border, 0.0)
    else:
        border = border_small

    base = np.where(label > 0, class_weight_fg, class_weight_bg).astype(np.float32)
    weight = np.maximum(base + border, 0.0).astype(np.float32)
    stats = {
        "objects": float(len([int(v) for v in np.unique(label) if int(v) != 0])),
        "fg_fraction": float(np.mean(label > 0)),
        "weight_min": float(np.min(weight)),
        "weight_max": float(np.max(weight)),
        "weight_mean": float(np.mean(weight)),
        "weight_std": float(np.std(weight)),
        "border_mean": float(np.mean(border)),
    }
    return weight, stats


def make_weight_map_exact(
    label: np.ndarray,
    w0: float,
    sigma: float,
    class_weight_bg: float,
    class_weight_fg: float,
    class_balance: str,
) -> tuple[np.ndarray, dict[str, float]]:
    from scipy import ndimage as ndi

    label = label.astype(np.int32, copy=False)
    ids = [int(v) for v in np.unique(label) if int(v) != 0]
    fg = label > 0
    fg_fraction = float(np.mean(fg))
    if class_balance == "inverse-frequency":
        eps = 1e-6
        class_weight_fg = 0.5 / max(fg_fraction, eps)
        class_weight_bg = 0.5 / max(1.0 - fg_fraction, eps)

    border = np.zeros(label.shape, dtype=np.float32)
    if len(ids) >= 2 and w0 > 0:
        d1 = np.full(label.shape, np.inf, dtype=np.float32)
        d2 = np.full(label.shape, np.inf, dtype=np.float32)
        structure = np.ones((3, 3), dtype=bool)
        for obj_id in ids:
            obj = label == obj_id
            eroded = ndi.binary_erosion(obj, structure=structure, border_value=0)
            boundary = obj & ~eroded
            if not np.any(boundary):
                boundary = obj
            dist = ndi.distance_transform_edt(~boundary).astype(np.float32)
            closer = dist < d1
            d2[closer] = d1[closer]
            d1[closer] = dist[closer]
            second = (~closer) & (dist < d2)
            d2[second] = dist[second]

        valid = np.isfinite(d1) & np.isfinite(d2)
        dsum = d1[valid] + d2[valid]
        border[valid] = w0 * np.exp(-(dsum * dsum) / (2.0 * sigma * sigma))

    base = np.where(fg, class_weight_fg, class_weight_bg).astype(np.float32)
    weight = np.maximum(base + border, 0.0).astype(np.float32)
    stats = {
        "objects": float(len(ids)),
        "fg_fraction": fg_fraction,
        "weight_min": float(np.min(weight)),
        "weight_max": float(np.max(weight)),
        "weight_mean": float(np.mean(weight)),
        "weight_std": float(np.std(weight)),
        "border_mean": float(np.mean(border)),
    }
    return weight, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Create U-Net style CTC border weight maps from instance masks.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="err_minus_gt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--w0", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--method", choices=["approximate", "exact"], default="approximate")
    parser.add_argument("--distance-scale", type=float, default=0.5)
    parser.add_argument("--max-boundary-points", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--class-weight-bg", type=float, default=1.0)
    parser.add_argument("--class-weight-fg", type=float, default=1.0)
    parser.add_argument("--class-balance", choices=["manual", "inverse-frequency"], default="manual")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not (0 < args.distance_scale <= 1):
        raise ValueError("--distance-scale must be in (0, 1]")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or (args.out_dir / "weight_map_summary.csv")

    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == args.split]
    if not rows:
        raise RuntimeError(f"No rows for split {args.split} in {args.manifest}")

    summary_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        out_file = args.out_dir / f"{sample_id}_weight.npy"
        if out_file.exists() and not args.overwrite:
            weight = np.load(out_file)
            label = np.asarray(Image.open(resolve_path(row["mask_src"], args.manifest)), dtype=np.int32)
            stats = {
                "objects": float(len([int(v) for v in np.unique(label) if int(v) != 0])),
                "fg_fraction": float(np.mean(label > 0)),
                "weight_min": float(np.min(weight)),
                "weight_max": float(np.max(weight)),
                "weight_mean": float(np.mean(weight)),
                "weight_std": float(np.std(weight)),
                "border_mean": float(np.mean(weight - np.where(label > 0, args.class_weight_fg, args.class_weight_bg))),
            }
        else:
            label = np.asarray(Image.open(resolve_path(row["mask_src"], args.manifest)), dtype=np.int32)
            if args.method == "exact":
                weight, stats = make_weight_map_exact(
                    label=label,
                    w0=args.w0,
                    sigma=args.sigma,
                    class_weight_bg=args.class_weight_bg,
                    class_weight_fg=args.class_weight_fg,
                    class_balance=args.class_balance,
                )
            else:
                weight, stats = make_weight_map(
                    label=label,
                    w0=args.w0,
                    sigma=args.sigma,
                    distance_scale=args.distance_scale,
                    max_boundary_points=args.max_boundary_points,
                    chunk_size=args.chunk_size,
                    class_weight_bg=args.class_weight_bg,
                    class_weight_fg=args.class_weight_fg,
                )
            np.save(out_file, weight)

        summary_rows.append(
            {
                "split": args.split,
                "sample_id": sample_id,
                "weight_path": str(out_file),
                "objects": f"{stats['objects']:.0f}",
                "fg_fraction": f"{stats['fg_fraction']:.8f}",
                "weight_min": f"{stats['weight_min']:.6f}",
                "weight_max": f"{stats['weight_max']:.6f}",
                "weight_mean": f"{stats['weight_mean']:.6f}",
                "weight_std": f"{stats['weight_std']:.6f}",
                "border_mean": f"{stats['border_mean']:.6f}",
                "method": args.method,
                "class_balance": args.class_balance,
            }
        )
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"{index}/{len(rows)} {sample_id}: max={stats['weight_max']:.3f} mean={stats['weight_mean']:.3f}")

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split",
            "sample_id",
            "weight_path",
            "objects",
            "fg_fraction",
            "weight_min",
            "weight_max",
            "weight_mean",
            "weight_std",
            "border_mean",
            "method",
            "class_balance",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {len(summary_rows)} weight maps to {args.out_dir}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
