"""Independent confirmation of the already-locked centered-mean selections.

This script never chooses a tau from the newly generated test data.  It reads
the frozen choices from the completed development run, evaluates the existing
checkpoints on a fresh test stream, and reports image-level MSE intervals and
both PSNR aggregations.  The old 2026-10-08 test scores are therefore kept as
development evidence, while this directory is the confirmation protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as student_t

from noise_models import make_pair, sample_observation
from run_allocation import BoundedCNN, HIGH, LOW, MODELS, SPLITS, predict, sample_images, score_arrays, seed_for

DATA_RANGE = HIGH - LOW
METHODS = ("centered_mean", "fixed_tau_1", "raw_J", "clean_val_oracle")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def image_interval(values: np.ndarray, delta: float = 0.05) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(n)
    radius = float(student_t.ppf(1 - delta / 2, n - 1) * se)
    # Empirical Bernstein for [0, DATA_RANGE**2], as a finite-sample
    # concentration interval.  This is conditional on the frozen models.
    log_term = np.log(4.0 / delta)
    eb_radius = np.sqrt(2 * sd * sd * log_term / n) + 7 * DATA_RANGE**2 * log_term / (3 * (n - 1))
    return {
        "mse": mean, "mse_sd": sd, "mse_se": se,
        "mse_t_low": mean - radius, "mse_t_high": mean + radius,
        "mse_eb_low": max(0.0, mean - eb_radius),
        "mse_eb_high": min(DATA_RANGE**2, mean + eb_radius),
        "n_images": int(n),
    }


def psnr_summary(mse_arrays: list[np.ndarray]) -> dict[str, float]:
    per_image = [10.0 * np.log10(DATA_RANGE**2 / a) for a in mse_arrays]
    flat_mse = np.concatenate(mse_arrays)
    flat_psnr = np.concatenate(per_image)
    return {
        "pooled_mse": float(flat_mse.mean()),
        "pooled_psnr_db": float(10.0 * np.log10(DATA_RANGE**2 / flat_mse.mean())),
        "mean_per_image_psnr_db": float(flat_psnr.mean()),
        "between_rep_psnr_sd_db": float(np.std([a.mean() for a in per_image], ddof=1)) if len(per_image) > 1 else 0.0,
    }


def main(args: argparse.Namespace) -> None:
    run = Path(args.run).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "scores").mkdir()
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    selections = read_rows(run / "selections.csv")
    params = {key: protocol[key] for key in ("gamma", "sigma", "ell")}
    models = protocol["models"]
    reps = int(protocol["reps"])
    taus = [float(t) for t in protocol["taus"]]
    # Convert the frozen development choices once, before drawing any new
    # test observations.  The test stream cannot affect these values.
    frozen: dict[tuple[str, int], dict[str, float]] = {}
    for model in models:
        for rep in range(reps):
            rows = [r for r in selections if r["model"] == model and int(r["rep"]) == rep]
            if len(rows) != len(set(r["method"] for r in rows)):
                raise ValueError(f"duplicate or missing frozen selections for {model}, rep {rep}")
            frozen[(model, rep)] = {r["method"]: float(r["tau"]) for r in rows}
            missing = set(METHODS) - set(frozen[(model, rep)])
            if missing:
                raise ValueError(f"missing frozen choices {missing} for {model}, rep {rep}")

    metadata = {
        "source_run": str(run), "source_selection_file": str(run / "selections.csv"),
        "selection_locked_before_test": True,
        "test_seed": args.test_seed, "repetitions": reps,
        "test_n": int(protocol["test_n"]), "size": int(protocol["size"]),
        "models": models, "methods": list(METHODS), "taus": taus,
        "data_range": DATA_RANGE,
        "estimand": "test risk conditional on the five frozen trained models; image-level intervals do not include training randomness",
        "psnr": "10 log10((2-0.25)^2 / per-image-MSE); pooled and mean-per-image are both reported",
        "test_stream": "fresh X,Y,Z,T generated after choices were read; no test quantity used for tau selection",
    }
    (out / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    all_scores: dict[tuple[str, str], list[np.ndarray]] = {(m, method): [] for m in models for method in METHODS}
    selection_rows, test_rows = [], []
    for model in models:
        model_id = MODELS[model]
        for rep in range(reps):
            choice = frozen[(model, rep)]
            union_taus = sorted(set(choice.values()) | {1.0})
            x = sample_images(
                int(protocol["test_n"]), int(protocol["size"]),
                np.random.default_rng(seed_for(args.test_seed, rep, SPLITS["test"])),
            )
            y = sample_observation(
                x, model,
                np.random.default_rng(seed_for(args.test_seed, rep, SPLITS["test"] + 100, model_id)),
                **params,
            )
            cache: dict[float, dict[str, np.ndarray]] = {}
            for tau in union_taus:
                # The tau-specific stream is deterministic but distinct; the
                # test data are still independent from the old development run.
                rng = np.random.default_rng(seed_for(
                    args.test_seed, rep, SPLITS["test"] + 200 + int(round(10 * tau)), model_id
                ))
                z, target = make_pair(
                    y, model, tau, rng,
                    quad_order=max(int(protocol["quad_order"]), 48) if model == "gg" and tau <= 0.25 else int(protocol["quad_order"]),
                    **params,
                )
                checkpoint = run / "checkpoints" / f"{model}_r{rep}_tau{tau:g}.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                net = BoundedCNN()
                net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                cache[tau] = score_arrays(predict(net, z), x, y, target, model, params)
                np.savez_compressed(out / "scores" / f"{model}_r{rep}_tau{tau:g}_test.npz", **cache[tau])
            for method in METHODS:
                tau = choice[method]
                clean = cache[tau]["clean"]
                all_scores[(model, method)].append(clean)
                selection_rows.append({"model": model, "rep": rep, "method": method, "tau": tau})
                row = {"model": model, "rep": rep, "method": method, "tau": tau}
                row.update(image_interval(clean))
                row.update(psnr_summary([clean]))
                test_rows.append(row)

    summaries = []
    for model in models:
        for method in METHODS:
            arrays = all_scores[(model, method)]
            flat = np.concatenate(arrays)
            row = {"model": model, "method": method, "tau_values": ",".join(f"{x:g}" for x in sorted(set(r["tau"] for r in selection_rows if r["model"] == model and r["method"] == method)))}
            row.update(image_interval(flat))
            row.update(psnr_summary(arrays))
            row["mean_selected_tau"] = float(np.mean([r["tau"] for r in selection_rows if r["model"] == model and r["method"] == method]))
            summaries.append(row)

    paired = []
    for model in models:
        base = all_scores[(model, "fixed_tau_1")]
        base_flat = np.concatenate(base)
        base_psnr = psnr_summary(base)
        for method in METHODS:
            current = all_scores[(model, method)]
            diff = np.concatenate(current) - base_flat
            row = {"model": model, "method": method, "comparison": "method_minus_fixed_tau_1"}
            row.update(image_interval(diff))
            row["mse_difference"] = float(diff.mean())
            row["psnr_difference_db"] = float(psnr_summary(current)["mean_per_image_psnr_db"] - base_psnr["mean_per_image_psnr_db"])
            paired.append(row)

    save_csv(out / "frozen_selections.csv", selection_rows)
    save_csv(out / "test_by_rep.csv", test_rows)
    save_csv(out / "test_summary.csv", summaries)
    save_csv(out / "paired_vs_fixed_tau_1.csv", paired)
    (out / "test_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (out / "paired_vs_fixed_tau_1.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "summary_rows": len(summaries), "test_rows": len(test_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="output/experiments/allocation_main_20261008")
    parser.add_argument("--out", default="output/experiments/centered_confirmation_20261009")
    parser.add_argument("--test-seed", type=int, default=20261009)
    main(parser.parse_args())
