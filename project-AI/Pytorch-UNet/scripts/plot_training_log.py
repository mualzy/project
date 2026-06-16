import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures/training_curves"))
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    text = args.log.read_text(encoding="utf-8", errors="ignore")
    dice = [float(x) for x in re.findall(r"Validation Dice score: ([0-9.eE+-]+)", text)]
    losses = [float(x) for x in re.findall(r"loss \(batch\)=([0-9.eE+-]+)", text)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if losses:
        plt.figure(figsize=(7, 4))
        plt.plot(losses)
        plt.xlabel("logged batch")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig(args.out_dir / f"{args.prefix}_loss_curve.png", dpi=160)
        plt.close()
    if dice:
        plt.figure(figsize=(7, 4))
        plt.plot(dice, marker="o")
        plt.xlabel("validation event")
        plt.ylabel("dice")
        plt.tight_layout()
        plt.savefig(args.out_dir / f"{args.prefix}_metric_curve.png", dpi=160)
        plt.close()
    print(f"loss_points={len(losses)} dice_points={len(dice)}")


if __name__ == "__main__":
    main()
