"""Create the publication figure for natural-image mixed-noise validation."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NOISES = ("pg", "gg", "gp")
LABELS = {"pg": "PG", "gg": "GG", "gp": "GP"}
COLORS = {"pg": "#0072B2", "gg": "#D55E00", "gp": "#009E73"}
DATASETS = ("B100", "Set14", "Urban100")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/experiments/natural_mixed_base_20261015/results/aggregate"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("output/figures/natural_mixed_dncnn_validation_external"),
    )
    args = parser.parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    candidates = read_csv(args.root / "all_candidate_rows.csv")
    paired = read_csv(args.root / "paired_vs_fixed.csv")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in candidates:
        grouped[(row["noise"], float(row["allocation"]))].append(float(row["val_q_mean"]))
    taus = np.asarray(sorted({tau for _, tau in grouped}), dtype=np.float64)
    for noise in NOISES:
        means = np.asarray([np.mean(grouped[(noise, tau)]) for tau in taus])
        sds = np.asarray([np.std(grouped[(noise, tau)], ddof=1) for tau in taus])
        axes[0].errorbar(
            taus,
            means,
            yerr=sds,
            color=COLORS[noise],
            marker="o",
            markersize=4,
            linewidth=1.3,
            capsize=2.5,
            label=LABELS[noise],
        )
        selected = int(np.argmin(means))
        axes[0].scatter(
            [taus[selected]],
            [means[selected]],
            marker="*",
            s=62,
            color=COLORS[noise],
            edgecolor="black",
            linewidth=0.35,
            zorder=4,
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(taus, [f"{value:g}" for value in taus])
    axes[0].set_xlabel(r"Allocation $\tau$")
    axes[0].set_ylabel(r"Validation centered $Q$ (mean $\pm$ SD)")
    axes[0].set_title("(a) Label-free validation curves")
    axes[0].grid(alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")

    x = np.arange(len(DATASETS), dtype=np.float64)
    width = 0.22
    for index, noise in enumerate(NOISES):
        rows = [
            next(
                row for row in paired
                if row["noise"] == noise and row["dataset"] == dataset and row["method"] == "q_r"
            )
            for dataset in DATASETS
        ]
        means = np.asarray([float(row["delta_pixel_pooled_psnr_mean"]) for row in rows])
        lows = np.asarray([float(row["delta_pixel_pooled_psnr_ci_low"]) for row in rows])
        highs = np.asarray([float(row["delta_pixel_pooled_psnr_ci_high"]) for row in rows])
        positions = x + (index - 1) * width
        axes[1].bar(positions, means, width=width, color=COLORS[noise], alpha=0.86, label=LABELS[noise])
        axes[1].errorbar(
            positions,
            means,
            yerr=np.vstack([means - lows, highs - means]),
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2.5,
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, DATASETS)
    axes[1].set_ylabel(r"Pooled-PSNR gain over fixed $\tau=1$ (dB)")
    axes[1].set_title("(b) External natural-image tests")
    axes[1].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[1].legend(frameon=False, ncol=3, loc="upper left")

    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(args.output_prefix.with_suffix(".pdf"))
    print(args.output_prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
