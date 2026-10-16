"""Summarize the matched supervised GR2R reference across training seeds.

The primary confidence intervals use the five independent training seeds as
the sampling units.  Images evaluated by the same trained model are not treated
as independent training repetitions.  Existing image-level intervals remain
conditional fixed-model summaries in each seed's ``test_results.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

T_975_DF4 = 2.7764451051977987


def read_rows(root: Path, pattern: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob(pattern)):
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, math.nan, math.nan
    half = T_975_DF4 * float(array.std(ddof=1)) / math.sqrt(len(array))
    return mean, mean - half, mean + half


def summarize_group(rows: list[dict[str, str]], source: str) -> dict:
    mse = [float(row["fresh_test_mc_mse"]) for row in rows]
    pooled_seed_psnr = [float(row["fresh_test_mc_pooled_psnr"]) for row in rows]
    image_psnr = [float(row["fresh_test_mc_mean_image_psnr"]) for row in rows]
    mse_mean, mse_low, mse_high = mean_ci(mse)
    seed_psnr_mean, seed_psnr_low, seed_psnr_high = mean_ci(pooled_seed_psnr)
    alphas = [float(row["alpha"]) for row in rows]
    counts = Counter(alphas)
    return {
        "source": source,
        "noise": rows[0]["noise"],
        "method": rows[0]["method"],
        "n_training_seeds": len(rows),
        "alpha_by_seed": ";".join(f"{float(row['alpha']):.2f}" for row in sorted(rows, key=lambda x: int(x["seed"]))),
        "alpha_mode": min(counts, key=lambda alpha: (-counts[alpha], alpha)),
        "test_mc_mse": mse_mean,
        "test_mc_mse_seed_ci_low": mse_low,
        "test_mc_mse_seed_ci_high": mse_high,
        "test_mc_pooled_psnr_from_aggregate_mse": 10 * math.log10(1 / mse_mean),
        "test_mc_mean_seed_pooled_psnr": seed_psnr_mean,
        "test_mc_mean_seed_pooled_psnr_ci_low": seed_psnr_low,
        "test_mc_mean_seed_pooled_psnr_ci_high": seed_psnr_high,
        "test_mc_mean_image_psnr": float(np.mean(image_psnr)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervised", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    supervised = read_rows(args.supervised, "seed*/test_results.csv")
    baseline = [
        row
        for row in read_rows(args.baseline, "seed*/fresh_test_results.csv")
        if row["method"] == "centered_R_mc"
    ]
    expected = 5 * 3 * 2
    if len(supervised) != expected:
        raise RuntimeError(f"expected {expected} supervised rows, found {len(supervised)}")
    if len(baseline) != 5 * 3:
        raise RuntimeError(f"expected 15 centered_R_mc rows, found {len(baseline)}")

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for source, rows in (("supervised", supervised), ("self_supervised", baseline)):
        for row in rows:
            grouped.setdefault((source, row["noise"], row["method"]), []).append(row)
    summaries = [
        summarize_group(rows, source)
        for (source, _, _), rows in sorted(grouped.items())
    ]

    deltas: list[dict] = []
    for noise in sorted({row["noise"] for row in supervised}):
        base_by_seed = {
            int(row["seed"]): row for row in baseline if row["noise"] == noise
        }
        for method in ("supervised_oracle_single", "supervised_oracle_mc"):
            sup_by_seed = {
                int(row["seed"]): row
                for row in supervised
                if row["noise"] == noise and row["method"] == method
            }
            seeds = sorted(set(base_by_seed) & set(sup_by_seed))
            mse_delta = [
                float(sup_by_seed[seed]["fresh_test_mc_mse"])
                - float(base_by_seed[seed]["fresh_test_mc_mse"])
                for seed in seeds
            ]
            psnr_delta = [
                float(sup_by_seed[seed]["fresh_test_mc_pooled_psnr"])
                - float(base_by_seed[seed]["fresh_test_mc_pooled_psnr"])
                for seed in seeds
            ]
            dm, dl, dh = mean_ci(mse_delta)
            pm, pl, ph = mean_ci(psnr_delta)
            deltas.append(
                {
                    "noise": noise,
                    "supervised_method": method,
                    "reference_method": "centered_R_mc",
                    "n_paired_training_seeds": len(seeds),
                    "mse_delta_supervised_minus_self": dm,
                    "mse_delta_seed_ci_low": dl,
                    "mse_delta_seed_ci_high": dh,
                    "pooled_psnr_delta_supervised_minus_self_db": pm,
                    "pooled_psnr_delta_seed_ci_low_db": pl,
                    "pooled_psnr_delta_seed_ci_high_db": ph,
                }
            )

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "supervised_reference_summary.csv", summaries)
    write_csv(args.out / "supervised_vs_centered_R_mc_deltas.csv", deltas)
    audit = {
        "supervised_rows": len(supervised),
        "baseline_rows": len(baseline),
        "confidence_interval_unit": "independent training seed (n=5)",
        "confidence_interval_method": "two-sided Student t interval with df=4",
        "fixed_model_intervals": "available separately in per-seed test_results.csv",
        "comparison": "paired by noise and training seed using the same fresh-test protocol",
        "interpretation": "finite-grid input-matched empirical supervised reference",
    }
    (args.out / "supervised_reference_audit.json").write_text(
        json.dumps(audit, indent=2)
    )
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
