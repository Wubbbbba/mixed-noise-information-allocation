"""Fresh-test comparison of frozen mixed-noise selections and supervision.

Both selection files are read and cryptographically recorded before any test
sample is drawn.  The fresh test set cannot alter tau or the comparison set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t as student_t

from noise_models import sample_observation
from run_allocation import BoundedCNN, HIGH, LOW, MODELS, SPLITS, predict, sample_images, seed_for
from run_mixed_supervised_reference import auxiliary_input


DATA_RANGE = HIGH - LOW
SELF_METHODS = ("centered_mean", "fixed_tau_1", "raw_J", "clean_val_oracle")
SUPERVISED_METHOD = "supervised_clean_train_val_select"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def ci_across_reps(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    se = sd / np.sqrt(a.size)
    radius = float(student_t.ppf(0.975, a.size - 1) * se)
    return {"mean": mean, "sd": sd, "se": se, "ci95_low": mean - radius, "ci95_high": mean + radius, "n_reps": int(a.size)}


def load_model(path: Path) -> BoundedCNN:
    net = BoundedCNN()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net


def main(args: argparse.Namespace) -> None:
    source = Path(args.run).resolve()
    supervised = Path(args.supervised_run).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "scores").mkdir()

    source_protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    completion = json.loads((supervised / "training_complete.json").read_text(encoding="utf-8"))
    if not completion.get("selection_locked_before_fresh_test"):
        raise ValueError("supervised selections were not locked")
    source_rows = read_csv(source / "selections.csv")
    supervised_rows = read_csv(supervised / "selections.csv")
    models = list(source_protocol["models"])
    reps = int(source_protocol["reps"])
    params = {k: float(source_protocol[k]) for k in ("gamma", "sigma", "ell")}

    choices: dict[tuple[str, int, str], tuple[float, str]] = {}
    for noise in models:
        for rep in range(reps):
            group = [r for r in source_rows if r["model"] == noise and int(r["rep"]) == rep]
            for method in SELF_METHODS:
                found = [r for r in group if r["method"] == method]
                if len(found) != 1:
                    raise ValueError(f"missing/duplicate {noise} r{rep} {method}")
                choices[(noise, rep, method)] = (float(found[0]["tau"]), "self_supervised")
            found = [r for r in supervised_rows if r["model"] == noise and int(r["rep"]) == rep and r["method"] == SUPERVISED_METHOD]
            if len(found) != 1:
                raise ValueError(f"missing/duplicate supervised selection {noise} r{rep}")
            choices[(noise, rep, SUPERVISED_METHOD)] = (float(found[0]["tau"]), "supervised")

    # This lock is written before the first call that generates fresh X or Y.
    frozen_rows = [
        {"model": noise, "rep": rep, "method": method, "tau": choices[(noise, rep, method)][0], "training": choices[(noise, rep, method)][1]}
        for noise in models for rep in range(reps) for method in (*SELF_METHODS, SUPERVISED_METHOD)
    ]
    save_csv(out / "frozen_selections_before_test.csv", frozen_rows)
    lock = {
        "selection_locked_before_test": True,
        "test_seed_chosen_before_test": int(args.test_seed),
        "source_selection_sha256": hashlib.sha256((source / "selections.csv").read_bytes()).hexdigest(),
        "supervised_selection_sha256": hashlib.sha256((supervised / "selections.csv").read_bytes()).hexdigest(),
        "frozen_snapshot_sha256": hashlib.sha256((out / "frozen_selections_before_test.csv").read_bytes()).hexdigest(),
        "methods": list((*SELF_METHODS, SUPERVISED_METHOD)),
    }
    (out / "selection_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol = {
        **lock,
        "source_run": str(source), "supervised_run": str(supervised),
        "test_n_per_rep": int(source_protocol["test_n"]), "reps": reps,
        "size": int(source_protocol["size"]), "models": models,
        "signal_range": [LOW, HIGH], "estimand": "f_tau(Z_tau)",
        "input_matching": "for a given noise, rep, and tau, every method receives the identical fresh X, Y, and Z_tau",
        "primary_interval": "two-sided t interval across five independent complete repetitions; includes training and finite-test variation represented by those repetitions",
        "psnr": "pooled = 10log10(range^2 / pooled MSE); mean-per-image averages image PSNR",
        "leakage_control": "all tau choices and methods frozen before fresh test generation; no test-based reselection",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    by_rep: list[dict] = []
    arrays: dict[tuple[str, str], list[np.ndarray]] = {(n, m): [] for n in models for m in (*SELF_METHODS, SUPERVISED_METHOD)}
    for noise in models:
        model_id = MODELS[noise]
        for rep in range(reps):
            x = sample_images(int(source_protocol["test_n"]), int(source_protocol["size"]), np.random.default_rng(seed_for(args.test_seed, rep, SPLITS["test"])))
            y = sample_observation(x, noise, np.random.default_rng(seed_for(args.test_seed, rep, SPLITS["test"] + 100, model_id)), **params)
            z_cache: dict[float, np.ndarray] = {}
            pred_cache: dict[tuple[str, float], np.ndarray] = {}
            for method in (*SELF_METHODS, SUPERVISED_METHOD):
                tau, training = choices[(noise, rep, method)]
                if tau not in z_cache:
                    z_cache[tau] = auxiliary_input(
                        y, noise, tau,
                        np.random.default_rng(seed_for(args.test_seed, rep, SPLITS["test"] + 200 + int(round(10 * tau)), model_id)),
                    )
                key = (training, tau)
                if key not in pred_cache:
                    base = supervised if training == "supervised" else source
                    checkpoint = base / "checkpoints" / f"{noise}_r{rep}_tau{tau:g}.pt"
                    if not checkpoint.exists():
                        raise FileNotFoundError(checkpoint)
                    pred_cache[key] = predict(load_model(checkpoint), z_cache[tau])
                mse = np.mean((pred_cache[key] - x) ** 2, axis=(1, 2, 3))
                image_psnr = 10.0 * np.log10(DATA_RANGE**2 / mse)
                arrays[(noise, method)].append(mse)
                row = {
                    "model": noise, "rep": rep, "method": method, "training": training, "tau": tau,
                    "test_mse": float(mse.mean()),
                    "pooled_psnr_db": float(10.0 * np.log10(DATA_RANGE**2 / mse.mean())),
                    "mean_per_image_psnr_db": float(image_psnr.mean()),
                    "test_mse_sd_across_images": float(mse.std(ddof=1)),
                    "n_test_images": int(mse.size),
                }
                by_rep.append(row)
                np.savez_compressed(out / "scores" / f"{noise}_r{rep}_{method}_tau{tau:g}_test.npz", clean_mse=mse, per_image_psnr_db=image_psnr)
                print(json.dumps({"event": "tested", "model": noise, "rep": rep, "method": method, "tau": tau, "mse": row["test_mse"]}), flush=True)

    save_csv(out / "test_by_rep.csv", by_rep)
    summaries: list[dict] = []
    for noise in models:
        for method in (*SELF_METHODS, SUPERVISED_METHOD):
            group = [r for r in by_rep if r["model"] == noise and r["method"] == method]
            mse_ci = ci_across_reps([r["test_mse"] for r in group])
            pooled_ci = ci_across_reps([r["pooled_psnr_db"] for r in group])
            image_ci = ci_across_reps([r["mean_per_image_psnr_db"] for r in group])
            flat = np.concatenate(arrays[(noise, method)])
            summaries.append({
                "model": noise, "method": method, "training": group[0]["training"],
                "tau_values": ",".join(f"{r['tau']:g}" for r in group),
                "mean_selected_tau": float(np.mean([r["tau"] for r in group])),
                "test_mse": mse_ci["mean"], "test_mse_ci95_low": mse_ci["ci95_low"], "test_mse_ci95_high": mse_ci["ci95_high"],
                "pooled_psnr_db": float(10.0 * np.log10(DATA_RANGE**2 / flat.mean())),
                "pooled_psnr_mean_across_reps_db": pooled_ci["mean"], "pooled_psnr_ci95_low_db": pooled_ci["ci95_low"], "pooled_psnr_ci95_high_db": pooled_ci["ci95_high"],
                "mean_per_image_psnr_db": float(np.mean(10.0 * np.log10(DATA_RANGE**2 / flat))),
                "mean_per_image_psnr_ci95_low_db": image_ci["ci95_low"], "mean_per_image_psnr_ci95_high_db": image_ci["ci95_high"],
                "n_reps": reps, "n_test_images_total": int(flat.size),
            })
    save_csv(out / "test_summary.csv", summaries)
    (out / "test_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    paired: list[dict] = []
    baselines = ("centered_mean", "fixed_tau_1", "raw_J", "clean_val_oracle")
    for noise in models:
        sup = [r for r in by_rep if r["model"] == noise and r["method"] == SUPERVISED_METHOD]
        for method in baselines:
            base = [r for r in by_rep if r["model"] == noise and r["method"] == method]
            base_by_rep = {int(r["rep"]): r for r in base}
            deltas_mse, deltas_pooled, deltas_image = [], [], []
            for s in sup:
                b = base_by_rep[int(s["rep"])]
                deltas_mse.append(float(s["test_mse"] - b["test_mse"]))
                deltas_pooled.append(float(s["pooled_psnr_db"] - b["pooled_psnr_db"]))
                deltas_image.append(float(s["mean_per_image_psnr_db"] - b["mean_per_image_psnr_db"]))
            mse_ci, pooled_delta_ci, image_delta_ci = ci_across_reps(deltas_mse), ci_across_reps(deltas_pooled), ci_across_reps(deltas_image)
            paired.append({
                "model": noise, "comparison": f"supervised_minus_{method}",
                "mse_difference": mse_ci["mean"], "mse_difference_ci95_low": mse_ci["ci95_low"], "mse_difference_ci95_high": mse_ci["ci95_high"],
                "pooled_psnr_difference_db": pooled_delta_ci["mean"], "pooled_psnr_difference_ci95_low_db": pooled_delta_ci["ci95_low"], "pooled_psnr_difference_ci95_high_db": pooled_delta_ci["ci95_high"],
                "mean_per_image_psnr_difference_db": image_delta_ci["mean"], "mean_per_image_psnr_difference_ci95_low_db": image_delta_ci["ci95_low"], "mean_per_image_psnr_difference_ci95_high_db": image_delta_ci["ci95_high"],
                "n_paired_reps": reps,
            })
    save_csv(out / "paired_supervised_comparisons.csv", paired)
    (out / "paired_supervised_comparisons.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "evaluation_complete.json").write_text(json.dumps({"rows": len(by_rep), "summaries": len(summaries), "paired": len(paired)}, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "out": str(out), "rows": len(by_rep)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="output/experiments/allocation_main_20261008")
    parser.add_argument("--supervised-run", default="output/experiments/mixed_supervised_reference_20261010")
    parser.add_argument("--out", default="output/experiments/mixed_supervised_confirmation_20261012")
    parser.add_argument("--test-seed", type=int, default=20261012)
    main(parser.parse_args())
