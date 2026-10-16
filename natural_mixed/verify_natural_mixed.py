"""Independent checks for the reported base-noise natural-image experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


NOISES = ("pg", "gg", "gp")
DATASETS = ("B100", "Set14", "Urban100")
METHODS = ("q_r", "raw_J", "fixed", "clean_val")
TAUS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DYNAMIC_RANGE = 1.75
T_CRITICAL_DF2 = 4.3026527297
TOL = 2e-10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(a: float, b: float, tolerance: float = TOL) -> bool:
    return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def seed_interval(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    radius = T_CRITICAL_DF2 * sd / math.sqrt(len(array))
    return mean, sd, mean - radius, mean + radius


def verify_run(
    run: Path,
    expected_seed: int,
    compact: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    assert complete == {"complete": True, "seed": expected_seed, "candidate_rows": 18, "test_rows": 36}
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["noise_level"] == "base"
    assert protocol["noise_parameters"] == {"gamma": 0.25, "sigma": 0.25, "ell": 4.0}
    assert protocol["epochs"] == 100 and protocol["train_count"] == 650
    assert protocol["models"] == list(NOISES) and protocol["modes"] == ["selfsup"]
    assert protocol["external_test"] == list(DATASETS)

    candidates = read_csv(run / "candidates.csv")
    tests = read_csv(run / "test_results.csv")
    assert len(candidates) == 18 and len(tests) == 36
    if not compact:
        assert len(list((run / "checkpoints").glob("*.pt"))) == 18
        assert len(list((run / "per_image").glob("*.npz"))) == 27

    selections: dict[str, dict[str, float]] = {}
    for noise in NOISES:
        rows = [row for row in candidates if row["noise"] == noise]
        assert sorted(float(row["allocation"]) for row in rows) == list(TAUS)
        selections[noise] = json.loads((run / f"{noise}_selection.json").read_text(encoding="utf-8"))
        assert float(selections[noise]["q_r"]) == float(min(rows, key=lambda row: float(row["val_q_mean"]))["allocation"])
        assert float(selections[noise]["raw_J"]) == float(min(rows, key=lambda row: float(row["val_raw_J"]))["allocation"])
        assert float(selections[noise]["clean_val"]) == float(min(rows, key=lambda row: float(row["val_clean_mse"]))["allocation"])
        assert float(selections[noise]["fixed"]) == 1.0
        if not compact:
            selection_time = (run / f"{noise}_selection.json").stat().st_mtime_ns
            test_times = [path.stat().st_mtime_ns for path in (run / "per_image").glob(f"{noise}_*.npz")]
            assert test_times and selection_time <= min(test_times)

    expected_keys = {(noise, method, dataset) for noise in NOISES for method in METHODS for dataset in DATASETS}
    observed_keys = {(row["noise"], row["method"], row["dataset"]) for row in tests}
    assert observed_keys == expected_keys

    for row in tests:
        noise, method, dataset = row["noise"], row["method"], row["dataset"]
        allocation = float(row["allocation"])
        assert close(allocation, float(selections[noise][method]))
        if not compact:
            archive = run / "per_image" / f"{noise}_selfsup_a{allocation:g}_{dataset}.npz"
            arrays = np.load(archive)
            mse = arrays["mse"].astype(np.float64)
            ssim = arrays["ssim"].astype(np.float64)
            assert len(mse) == int(row["n_images"])
            assert close(float(mse.mean()), float(row["image_mean_mse"]))
            assert close(float(np.mean(10 * np.log10(DYNAMIC_RANGE**2 / np.maximum(mse, 1e-12)))), float(row["mean_image_psnr"]))
            assert close(float(ssim.mean()), float(row["mean_image_ssim"]))
            assert close(float(10 * np.log10(DYNAMIC_RANGE**2 / mse.mean())), float(row["equal_image_pooled_psnr"]))

    return candidates, tests


def verify_aggregate(root: Path, candidates: list[dict[str, str]], tests: list[dict[str, str]]) -> None:
    aggregate = root / "aggregate"
    aggregate_rows = read_csv(aggregate / "test_seed_aggregate.csv")
    paired_rows = read_csv(aggregate / "paired_vs_fixed.csv")
    assert len(aggregate_rows) == 36 and len(paired_rows) == 27

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in tests:
        grouped[(row["noise"], row["dataset"], row["method"])].append(row)
    for row in aggregate_rows:
        key = (row["noise"], row["dataset"], row["method"])
        values = [float(item["pixel_pooled_psnr"]) for item in grouped[key]]
        mean, sd, low, high = seed_interval(values)
        assert close(mean, float(row["pixel_pooled_psnr_mean"]))
        assert close(sd, float(row["pixel_pooled_psnr_seed_sd"]))
        assert close(low, float(row["pixel_pooled_psnr_ci_low"]))
        assert close(high, float(row["pixel_pooled_psnr_ci_high"]))

    lookup = {(row["noise"], row["dataset"], row["method"], row["seed"]): row for row in tests}
    for row in paired_rows:
        noise, dataset, method = row["noise"], row["dataset"], row["method"]
        differences = [
            float(lookup[(noise, dataset, method, str(seed))]["pixel_pooled_psnr"])
            - float(lookup[(noise, dataset, "fixed", str(seed))]["pixel_pooled_psnr"])
            for seed in range(3)
        ]
        mean, sd, low, high = seed_interval(differences)
        assert close(mean, float(row["delta_pixel_pooled_psnr_mean"]))
        assert close(sd, float(row["delta_pixel_pooled_psnr_seed_sd"]))
        assert close(low, float(row["delta_pixel_pooled_psnr_ci_low"]))
        assert close(high, float(row["delta_pixel_pooled_psnr_ci_high"]))

    for noise in NOISES:
        for seed in range(3):
            rows = [row for row in candidates if row["noise"] == noise and int(row["seed"]) == seed]
            assert min(rows, key=lambda row: float(row["val_q_mean"]))["allocation"] == min(rows, key=lambda row: float(row["val_clean_mse"]))["allocation"]


def verify_aggregate_only(root: Path) -> None:
    aggregate = root / "aggregate"
    candidates = read_csv(aggregate / "all_candidate_rows.csv")
    tests = read_csv(aggregate / "all_test_rows.csv")
    assert len(candidates) == 54 and len(tests) == 108
    assert {int(row["seed"]) for row in candidates} == {0, 1, 2}
    assert {int(row["seed"]) for row in tests} == {0, 1, 2}
    verify_aggregate(root, candidates, tests)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/experiments/natural_mixed_base_20261015/results"),
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="verify the released CSV/JSON summaries without checkpoints or per-image arrays",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="verify public aggregate tables when seed directories are not installed",
    )
    args = parser.parse_args()
    if args.aggregate_only:
        verify_aggregate_only(args.root)
        print(json.dumps({
            "verification_mode": "aggregate-only",
            "candidate_models": 54,
            "test_rows": 108,
            "selection_matches_clean_validation": True,
            "verification": "passed",
        }, indent=2))
        return
    all_candidates: list[dict[str, str]] = []
    all_tests: list[dict[str, str]] = []
    for seed in range(3):
        candidates, tests = verify_run(args.root / f"base_seed{seed}", seed, args.compact)
        all_candidates.extend(candidates)
        all_tests.extend(tests)
    verify_aggregate(args.root, all_candidates, all_tests)
    print(json.dumps({
        "verification_mode": "compact" if args.compact else "full",
        "complete_runs": 3,
        "candidate_models": 54,
        "test_rows": 108,
        "per_image_archives_checked": 0 if args.compact else 81,
        "selection_matches_clean_validation": True,
        "verification": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
