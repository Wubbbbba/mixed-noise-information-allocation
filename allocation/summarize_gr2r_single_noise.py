"""Audit and summarize the five-seed single-noise GR2R comparison.

This script performs no training and no parameter selection.  It reads the
validation-locked choices and the fresh-test per-image arrays, then writes
CSV/JSON summaries next to them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


NOISES = ("gaussian", "poisson", "gamma")
METHODS = (
    # Historical result files use ``centered_R_mc`` as a schema key.  It is
    # retained only for backward compatibility; the selected score is Q_MC.
    "centered_R_mc",
    "paper_sensitivity",
    "paper_main",
    "raw_J",
    "clean_oracle_mc",
)
DISPLAY = {
    "centered_R_mc": "Q_MC selection",
    "paper_sensitivity": "GR2R sensitivity",
    "paper_main": "GR2R main",
    "raw_J": "raw J_stab",
    "clean_oracle_mc": "clean validation oracle",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alpha_text(values: list[float]) -> str:
    return ",".join(f"{value:g}" for value in values)


def main(run: Path) -> None:
    run = run.resolve()
    selections: dict[tuple[str, str], list[float]] = {
        (noise, method): [] for noise in NOISES for method in METHODS
    }
    arrays: dict[tuple[str, str], list[np.ndarray]] = {
        (noise, method): [] for noise in NOISES for method in METHODS
    }
    source_hashes: dict[str, str] = {}

    for seed in range(5):
        seed_dir = run / f"seed{seed}"
        complete_path = seed_dir / "complete.json"
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        assert complete["complete"] is True
        assert complete["training_performed"] is False
        assert complete["validation_selection_only"] is True
        source_hashes[str(complete_path.relative_to(run))] = sha256(complete_path)

        for noise in NOISES:
            selection_path = seed_dir / f"{noise}_selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            assert selection["selection_locked_before_fresh_test"] is True
            assert selection["seed"] == seed and selection["noise"] == noise
            source_hashes[str(selection_path.relative_to(run))] = sha256(selection_path)

            for method in METHODS:
                alpha = float(selection["selections"][method])
                score_path = seed_dir / "per_image" / f"{noise}_a{alpha:.2f}.npz"
                assert score_path.exists(), score_path
                with np.load(score_path) as score_file:
                    mse = np.asarray(score_file["mc_mse"], dtype=np.float64)
                assert mse.ndim == 1 and mse.size == 75
                assert np.all(np.isfinite(mse)) and np.all(mse > 0)
                selections[(noise, method)].append(alpha)
                arrays[(noise, method)].append(mse)
                source_hashes[str(score_path.relative_to(run))] = sha256(score_path)

    alpha_rows: list[dict] = []
    psnr_rows: list[dict] = []
    for noise in NOISES:
        for method in METHODS:
            alphas = selections[(noise, method)]
            counts = Counter(alphas)
            mode_count = max(counts.values())
            modes = sorted(alpha for alpha, count in counts.items() if count == mode_count)
            alpha_rows.append({
                "noise": noise,
                "method": DISPLAY[method],
                "alpha_by_seed": alpha_text(alphas),
                "mode_alpha": ",".join(f"{value:g}" for value in modes),
                "mode_count": mode_count,
                "selection_source": "validation" if method in {"centered_R_mc", "clean_oracle_mc"} else "fixed protocol",
            })

            mse = np.concatenate(arrays[(noise, method)])
            pooled_mse = float(mse.mean())
            pooled_psnr = float(10.0 * np.log10(1.0 / pooled_mse))
            mean_image_psnr = float(np.mean(10.0 * np.log10(1.0 / mse)))
            psnr_rows.append({
                "noise": noise,
                "method": DISPLAY[method],
                "alpha_by_seed": alpha_text(alphas),
                "n_images": int(mse.size),
                "pooled_mse": pooled_mse,
                "pooled_psnr_db": pooled_psnr,
                "mean_per_image_psnr_db": mean_image_psnr,
            })

    lookup = {(row["noise"], row["method"]): row for row in psnr_rows}
    delta_rows: list[dict] = []
    for noise in NOISES:
        selected = lookup[(noise, "Q_MC selection")]
        for baseline in ("GR2R sensitivity", "GR2R main"):
            base = lookup[(noise, baseline)]
            delta_rows.append({
                "noise": noise,
                "comparison": f"Q_MC selection minus {baseline}",
                "delta_pooled_psnr_db": selected["pooled_psnr_db"] - base["pooled_psnr_db"],
                "delta_mean_per_image_psnr_db": selected["mean_per_image_psnr_db"] - base["mean_per_image_psnr_db"],
            })

    write_csv(run / "alpha_correspondence.csv", alpha_rows)
    write_csv(run / "psnr_comparison.csv", psnr_rows)
    write_csv(run / "psnr_deltas.csv", delta_rows)
    audit = {
        "source_description": "validation-locked five-seed GR2R run",
        "training_performed_by_this_script": False,
        "selection_performed_by_this_script": False,
        "selections_locked_before_fresh_test": True,
        "aggregation": "concatenate five seeds of 75 fresh-test images; use finite-K MC MSE",
        "psnr_dynamic_range": 1.0,
        "source_sha256": source_hashes,
        "alpha_correspondence": alpha_rows,
        "psnr_comparison": psnr_rows,
        "psnr_deltas": delta_rows,
    }
    (run / "audit_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for row in psnr_rows:
        print(
            f'{row["noise"]:8s} {row["method"]:24s} '
            f'pooled={row["pooled_psnr_db"]:.3f} '
            f'mean-image={row["mean_per_image_psnr_db"]:.3f}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    main(parser.parse_args().run)
