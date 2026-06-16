import argparse
import csv
from collections import deque
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


def label_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = mask.astype(bool, copy=False)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    sizes = [0]
    current = 0

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            q: deque[tuple[int, int]] = deque()
            q.append((y, x))
            labels[y, x] = current
            size = 0
            while q:
                cy, cx = q.popleft()
                size += 1
                if cy > 0 and mask[cy - 1, cx] and labels[cy - 1, cx] == 0:
                    labels[cy - 1, cx] = current
                    q.append((cy - 1, cx))
                if cy + 1 < h and mask[cy + 1, cx] and labels[cy + 1, cx] == 0:
                    labels[cy + 1, cx] = current
                    q.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and labels[cy, cx - 1] == 0:
                    labels[cy, cx - 1] = current
                    q.append((cy, cx - 1))
                if cx + 1 < w and mask[cy, cx + 1] and labels[cy, cx + 1] == 0:
                    labels[cy, cx + 1] = current
                    q.append((cy, cx + 1))
            sizes.append(size)

    return labels, np.asarray(sizes, dtype=np.int64)


def ctc_seg_for_image(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, int, int]:
    reference = reference.astype(np.int32, copy=False)
    pred_labels, pred_sizes = label_components(prediction > 0)
    ref_ids = [int(v) for v in np.unique(reference) if int(v) != 0]
    if not ref_ids:
        return 0.0, 0, int(pred_sizes.shape[0] - 1)

    scores = []
    for ref_id in ref_ids:
        ref_mask = reference == ref_id
        ref_area = int(ref_mask.sum())
        overlapping = pred_labels[ref_mask]
        counts = np.bincount(overlapping.ravel(), minlength=pred_sizes.shape[0])
        counts[0] = 0
        pred_id = int(np.argmax(counts))
        intersection = int(counts[pred_id]) if pred_id > 0 else 0
        if pred_id == 0 or intersection <= 0.5 * ref_area:
            scores.append(0.0)
            continue
        union = ref_area + int(pred_sizes[pred_id]) - intersection
        scores.append(intersection / union if union else 0.0)
    return float(np.mean(scores)), len(ref_ids), int(pred_sizes.shape[0] - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CTC SEG from instance GT masks and binary prediction masks.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="gt_seg")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == args.split]
    if not rows:
        raise RuntimeError(f"No rows for split {args.split} in {args.manifest}")

    out_rows = []
    total_ref_objects = 0
    weighted_sum = 0.0
    for row in rows:
        sample_id = row["sample_id"]
        ref = np.asarray(Image.open(resolve_path(row["mask_src"], args.manifest)), dtype=np.int32)
        pred_path = args.pred_dir / f"{sample_id}.png"
        pred = np.asarray(Image.open(pred_path).convert("L").resize((ref.shape[1], ref.shape[0]), resample=Image.NEAREST))
        seg, ref_objects, pred_objects = ctc_seg_for_image(ref, pred)
        weighted_sum += seg * ref_objects
        total_ref_objects += ref_objects
        out_rows.append(
            {
                "sample_id": sample_id,
                "seg": f"{seg:.8f}",
                "reference_objects": ref_objects,
                "predicted_objects": pred_objects,
            }
        )

    mean_seg = weighted_sum / total_ref_objects if total_ref_objects else 0.0
    for row in out_rows:
        row["mean_seg"] = f"{mean_seg:.8f}"
        row["total_reference_objects"] = total_ref_objects

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "seg", "reference_objects", "predicted_objects", "mean_seg", "total_reference_objects"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"frames={len(out_rows)} reference_objects={total_ref_objects} mean_seg={mean_seg:.6f} out={args.out}")


if __name__ == "__main__":
    main()
