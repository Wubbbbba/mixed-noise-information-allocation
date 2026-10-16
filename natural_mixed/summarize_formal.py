"""Aggregate the reported natural-image experiment across training seeds."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


T_CRITICAL_DF2 = 4.3026527297
GROUP_KEYS = ["noise_level", "noise", "dataset", "method"]
METRICS = [
    "image_mean_mse",
    "pixel_pooled_mse",
    "pixel_pooled_psnr",
    "mean_image_psnr",
    "mean_image_ssim",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def interval(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, math.nan, math.nan, math.nan
    sd = float(array.std(ddof=1))
    critical = T_CRITICAL_DF2 if len(array) == 3 else 1.96
    radius = critical * sd / math.sqrt(len(array))
    return mean, sd, mean - radius, mean + radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / "aggregate"
    output.mkdir(exist_ok=True)

    test_rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []
    run_status: list[dict] = []
    for run in sorted(path for path in root.iterdir() if path.is_dir() and "_seed" in path.name):
        complete = run / "complete.json"
        run_status.append({"run": run.name, "complete": complete.exists()})
        if not complete.exists():
            continue
        test_rows.extend(read_rows(run / "test_results.csv"))
        candidate_rows.extend(read_rows(run / "candidates.csv"))

    (output / "run_status.json").write_text(json.dumps(run_status, indent=2), encoding="utf-8")
    if not test_rows:
        print(json.dumps({"complete_runs": 0, "message": "no completed runs yet"}))
        return

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in test_rows:
        grouped[tuple(row[key] for key in GROUP_KEYS)].append(row)

    aggregate_rows: list[dict] = []
    for key, rows in sorted(grouped.items()):
        result = dict(zip(GROUP_KEYS, key))
        result["seeds"] = ",".join(sorted(row["seed"] for row in rows))
        result["n_seeds"] = len(rows)
        result["selected_tau_by_seed"] = ",".join(row["allocation"] for row in sorted(rows, key=lambda r: int(r["seed"])))
        for metric in METRICS:
            mean, sd, low, high = interval([float(row[metric]) for row in rows])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_seed_sd"] = sd
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        aggregate_rows.append(result)
    write_rows(output / "test_seed_aggregate.csv", aggregate_rows)

    # Paired seed-level method differences relative to fixed tau=1.
    lookup = {
        (row["noise_level"], row["noise"], row["dataset"], row["method"], row["seed"]): row
        for row in test_rows
    }
    paired_rows: list[dict] = []
    for group in aggregate_rows:
        method = group["method"]
        if method == "fixed":
            continue
        prefix = (group["noise_level"], group["noise"], group["dataset"])
        available = sorted(
            seed for level, noise, dataset, candidate_method, seed in lookup
            if (level, noise, dataset) == prefix and candidate_method == method
            and (*prefix, "fixed", seed) in lookup
        )
        result = {**dict(zip(GROUP_KEYS, (*prefix, method))), "reference": "fixed_tau_1", "seeds": ",".join(available)}
        for metric in ["pixel_pooled_psnr", "mean_image_psnr", "mean_image_ssim", "pixel_pooled_mse"]:
            differences = [
                float(lookup[(*prefix, method, seed)][metric])
                - float(lookup[(*prefix, "fixed", seed)][metric])
                for seed in available
            ]
            mean, sd, low, high = interval(differences)
            result[f"delta_{metric}_mean"] = mean
            result[f"delta_{metric}_seed_sd"] = sd
            result[f"delta_{metric}_ci_low"] = low
            result[f"delta_{metric}_ci_high"] = high
        paired_rows.append(result)
    write_rows(output / "paired_vs_fixed.csv", paired_rows)

    selection_rows: list[dict] = []
    for key, rows in sorted(grouped.items()):
        counts = Counter(row["allocation"] for row in rows)
        selection_rows.append({
            **dict(zip(GROUP_KEYS, key)),
            "selection_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
        })
    write_rows(output / "selection_frequency.csv", selection_rows)
    write_rows(output / "all_test_rows.csv", test_rows)
    write_rows(output / "all_candidate_rows.csv", candidate_rows)
    print(json.dumps({"complete_runs": sum(item["complete"] for item in run_status), "aggregate_groups": len(aggregate_rows)}))


if __name__ == "__main__":
    main()
