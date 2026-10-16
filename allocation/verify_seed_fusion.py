"""Independent arithmetic checks for the frozen-seed fusion experiment."""

from __future__ import annotations

import csv
import math
import argparse
from pathlib import Path

import numpy as np


TOL = 1e-14


def read_rows(result_dir: Path, name: str) -> list[dict[str, str]]:
    with (result_dir / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main(result_dir: Path) -> None:
    arrays = np.load(result_dir / "test_per_image.npz")

    summary_ok = True
    for row in read_rows(result_dir, "test_summary.csv"):
        values = arrays[f"{row['noise']}_{row['method']}_mse"]
        mean = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(values.size))
        summary_ok &= abs(mean - float(row["mse"])) < TOL
        summary_ok &= abs(se - float(row["mse_se"])) < TOL
        print(row["noise"], row["method"], f"mean={mean:.12f}", f"se={se:.12f}")

    paired_ok = True
    for row in read_rows(result_dir, "paired_differences.csv"):
        method = arrays[f"{row['noise']}_{row['method']}_mse"]
        baseline = arrays[f"{row['noise']}_{row['baseline']}_mse"]
        differences = method - baseline
        mean = float(differences.mean())
        se = float(differences.std(ddof=1) / math.sqrt(differences.size))
        paired_ok &= abs(mean - float(row["mse_difference"])) < TOL
        paired_ok &= abs(se - float(row["mse_difference_se"])) < TOL

    print(f"SUMMARY_RECOMPUTATION_OK={summary_ok}")
    print(f"PAIRED_RECOMPUTATION_OK={paired_ok}")
    if not (summary_ok and paired_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("output/experiments/seed_fusion_20261015"),
    )
    main(parser.parse_args().result_dir)
