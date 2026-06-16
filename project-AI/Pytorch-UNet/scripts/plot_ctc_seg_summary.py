import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("results/tables/ctc_seg_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/figures/ctc_seg_summary.png"))
    args = parser.parse_args()

    with args.summary.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labels = [row["experiment_id"].split("_")[0] for row in rows]
    values = [float(row["mean_seg"]) for row in rows]
    colors = ["#4477AA" if row["dataset"].startswith("DIC") else "#228833" for row in rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.bar(range(len(values)), values, color=colors)
    plt.xticks(range(len(values)), labels, rotation=35, ha="right")
    plt.ylabel("CTC SEG")
    plt.ylim(0, max(values) * 1.2 if values else 1)
    for i, value in enumerate(values):
        plt.text(i, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    plt.close()
    print(args.out)


if __name__ == "__main__":
    main()
