"""Compare the existing Gamma--Poisson Student-t construction with NB thinning.

The thinning candidates are selected on the frozen validation protocol.  A
selection-lock file is written before a new confirmation test set is generated.
Existing Student-t checkpoints are reused; only the thinning models are trained.
"""
from __future__ import annotations
# LEGACY NOTATION: locked artifacts use field name ``r`` for the GP thinning
# proportion denoted by pi in the manuscript.  Keep the serialized schema and
# checkpoint names unchanged so existing selection locks and hashes remain valid.


import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
import scipy
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from noise_models import make_pair, sample_observation  # noqa: E402
from run_allocation import (  # noqa: E402
    BoundedCNN,
    HIGH,
    LOW,
    MODELS,
    SPLITS,
    predict,
    sample_images,
    seed_for,
    train,
)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def thinning_pair(
    y: np.ndarray,
    r: float,
    gamma: float,
    ell: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Beta-binomial split of a NB count, rescaled to two unbiased views."""
    if not 0.0 < r < 1.0:
        raise ValueError("r must lie strictly between zero and one")
    scaled = y / gamma
    counts = np.rint(scaled).astype(np.int64)
    if np.any(counts < 0) or not np.allclose(scaled, counts, rtol=1e-10, atol=1e-7):
        raise ValueError("Gamma--Poisson observations must be gamma times counts")
    split_probability = rng.beta(r * ell, (1.0 - r) * ell, size=counts.shape)
    first = rng.binomial(counts, split_probability)
    second = counts - first
    return gamma * first / r, gamma * second / (1.0 - r)


def image_metrics(pred: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axes = tuple(range(1, pred.ndim))
    mse = np.mean((pred - clean) ** 2, axis=axes)
    psnr = 10.0 * np.log10((HIGH - LOW) ** 2 / mse)
    return mse, psnr


def mean_t_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if values.size < 2:
        return mean, math.nan, math.nan
    radius = float(student_t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size))
    return mean, mean - radius, mean + radius


def state_path(source: Path, rep: int, tau: float) -> Path:
    return source / "checkpoints" / f"gp_r{rep}_tau{tau:g}.pt"


def load_model(path: Path) -> BoundedCNN:
    model = BoundedCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def run(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    (out / "scores").mkdir()

    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    source_selections = load_csv(source / "selections.csv")
    required = {
        "reps": 5,
        "train_n": 384,
        "val_n": 512,
        "test_n": 2048,
        "size": 16,
        "epochs": 40,
        "batch_size": 64,
        "lr": 0.002,
        "gamma": 0.25,
        "ell": 4.0,
    }
    for key, expected in required.items():
        if protocol[key] != expected:
            raise ValueError(f"Unexpected source protocol {key}={protocol[key]!r}; expected {expected!r}")

    torch.set_num_threads(int(protocol["threads"]))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    train_args = argparse.Namespace(
        lr=float(protocol["lr"]),
        epochs=int(protocol["epochs"]),
        batch_size=int(protocol["batch_size"]),
    )
    gamma, ell = float(protocol["gamma"]), float(protocol["ell"])
    params = {"gamma": gamma, "sigma": float(protocol["sigma"]), "ell": ell}
    rs = [float(value) for value in args.r_values]

    manifest = {
        "purpose": "Gamma--Poisson Student-t versus negative-binomial thinning",
        "source": str(source),
        "source_protocol_sha256": hashlib.sha256((source / "protocol.json").read_bytes()).hexdigest(),
        "r_values": rs,
        "validation_base_seed": int(protocol["seed"]),
        "confirmation_test_seed": int(args.test_seed),
        "test_generated_after_selection_lock": True,
        "reps": int(protocol["reps"]),
        "train_n": int(protocol["train_n"]),
        "val_n": int(protocol["val_n"]),
        "test_n": int(args.test_n),
        "network": "same BoundedCNN as allocation_main_20261008",
        "training_budget": {"epochs": train_args.epochs, "batch_size": train_args.batch_size, "lr": train_args.lr},
        "risk": "single randomized input f(Z), common confirmation X/Y across methods within each seed",
        "interval_unit": "five complete training seeds",
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    candidate_rows: list[dict] = []
    choice_rows: list[dict] = []
    thinning_models: dict[tuple[int, float], Path] = {}
    for rep in range(int(protocol["reps"])):
        x_train = sample_images(int(protocol["train_n"]), int(protocol["size"]),
                                np.random.default_rng(seed_for(int(protocol["seed"]), rep, SPLITS["train"])))
        x_val = sample_images(int(protocol["val_n"]), int(protocol["size"]),
                              np.random.default_rng(seed_for(int(protocol["seed"]), rep, SPLITS["val"])))
        y_train = sample_observation(
            x_train, "gp", np.random.default_rng(seed_for(int(protocol["seed"]), rep, SPLITS["train"] + 100, MODELS["gp"])), **params
        )
        y_val = sample_observation(
            x_val, "gp", np.random.default_rng(seed_for(int(protocol["seed"]), rep, SPLITS["val"] + 100, MODELS["gp"])), **params
        )
        local: list[dict] = []
        for index, r in enumerate(rs):
            z_train, t_train = thinning_pair(
                y_train, r, gamma, ell,
                np.random.default_rng(seed_for(int(protocol["seed"]), rep, 400 + index, MODELS["gp"])),
            )
            z_val, t_val = thinning_pair(
                y_val, r, gamma, ell,
                np.random.default_rng(seed_for(int(protocol["seed"]), rep, 500 + index, MODELS["gp"])),
            )
            model, history = train(
                z_train, t_train, train_args, seed_for(int(protocol["seed"]), rep, 777)
            )
            pred = predict(model, z_val)
            axes = tuple(range(1, pred.ndim))
            centered = np.mean(pred**2 - 2.0 * pred * t_val, axis=axes)
            clean = np.mean((pred - x_val) ** 2, axis=axes)
            raw = np.mean((pred - t_val) ** 2, axis=axes)
            row = {
                "rep": rep,
                "r": r,
                "val_centered_mean": float(centered.mean()),
                "val_clean_mse_diagnostic": float(clean.mean()),
                "val_raw_mean": float(raw.mean()),
                "val_raw_sd": float(raw.std(ddof=1)),
                "train_raw_loss_final": float(history[-1]),
                "max_abs_train_target": float(np.max(np.abs(t_train))),
                "max_abs_val_target": float(np.max(np.abs(t_val))),
            }
            candidate_rows.append(row)
            local.append(row)
            checkpoint = out / "checkpoints" / f"gp_thinning_r{rep}_frac{r:g}.pt"
            torch.save(model.state_dict(), checkpoint)
            thinning_models[(rep, r)] = checkpoint
            np.savez_compressed(out / "scores" / f"gp_thinning_r{rep}_frac{r:g}_val.npz",
                                centered=centered, clean=clean, raw=raw)
            print(json.dumps({"event": "candidate", **row}), flush=True)

        choices = {
            "thinning_centered": min(local, key=lambda row: row["val_centered_mean"])["r"],
            "thinning_clean_oracle": min(local, key=lambda row: row["val_clean_mse_diagnostic"])["r"],
            "thinning_fixed_r05": min(rs, key=lambda value: abs(value - 0.5)),
        }
        for method, r in choices.items():
            source_key = "val_centered_mean" if method == "thinning_centered" else "val_clean_mse_diagnostic"
            if method == "thinning_fixed_r05":
                source_key = "fixed"
            objective = next((row[source_key] for row in local if row["r"] == r), math.nan) if source_key != "fixed" else math.nan
            choice_rows.append({"rep": rep, "method": method, "parameter": r, "parameter_name": "r", "val_objective": objective})

        for method, source_method in (
            ("student_centered", "centered_mean"),
            ("student_clean_oracle", "clean_val_oracle"),
            ("student_fixed_tau1", "fixed_tau_1"),
        ):
            matches = [row for row in source_selections
                       if row["model"] == "gp" and int(float(row["rep"])) == rep and row["method"] == source_method]
            if len(matches) != 1:
                raise ValueError(f"Missing unique source selection for rep={rep}, method={source_method}")
            choice_rows.append({"rep": rep, "method": method, "parameter": float(matches[0]["tau"]),
                                "parameter_name": "tau", "val_objective": math.nan})

    save_csv(out / "candidates.csv", candidate_rows)
    save_csv(out / "selection_lock.csv", choice_rows)
    lock_payload = {
        "locked_before_confirmation_test": True,
        "choices": choice_rows,
        "candidate_sha256": hashlib.sha256((out / "candidates.csv").read_bytes()).hexdigest(),
    }
    (out / "selection_lock.json").write_text(json.dumps(lock_payload, indent=2), encoding="utf-8")

    test_rows: list[dict] = []
    per_image: dict[str, np.ndarray] = {}
    for rep in range(int(protocol["reps"])):
        x_test = sample_images(int(args.test_n), int(protocol["size"]),
                               np.random.default_rng(seed_for(int(args.test_seed), rep, SPLITS["test"])))
        y_test = sample_observation(
            x_test, "gp", np.random.default_rng(seed_for(int(args.test_seed), rep, SPLITS["test"] + 100, MODELS["gp"])), **params
        )
        for choice in [row for row in choice_rows if row["rep"] == rep]:
            method = str(choice["method"])
            parameter = float(choice["parameter"])
            if method.startswith("thinning_"):
                index = rs.index(parameter)
                z_test, _ = thinning_pair(
                    y_test, parameter, gamma, ell,
                    np.random.default_rng(seed_for(int(args.test_seed), rep, 600 + index, MODELS["gp"])),
                )
                model = load_model(thinning_models[(rep, parameter)])
            else:
                z_test, _ = make_pair(
                    y_test, "gp", parameter,
                    np.random.default_rng(seed_for(int(args.test_seed), rep, 700, MODELS["gp"])), **params
                )
                model = load_model(state_path(source, rep, parameter))
            mse, psnr = image_metrics(predict(model, z_test), x_test)
            pooled_psnr = float(10.0 * np.log10((HIGH - LOW) ** 2 / mse.mean()))
            row = {
                "rep": rep,
                "method": method,
                "parameter": parameter,
                "parameter_name": choice["parameter_name"],
                "test_mse": float(mse.mean()),
                "pooled_psnr": pooled_psnr,
                "mean_image_psnr": float(psnr.mean()),
                "test_n": int(args.test_n),
            }
            test_rows.append(row)
            per_image[f"{method}_rep{rep}_mse"] = mse
            per_image[f"{method}_rep{rep}_psnr"] = psnr
            print(json.dumps({"event": "test", **row}), flush=True)

    save_csv(out / "test_by_seed.csv", test_rows)
    np.savez_compressed(out / "test_per_image.npz", **per_image)

    summary_rows: list[dict] = []
    methods = sorted({row["method"] for row in test_rows})
    for method in methods:
        rows = sorted((row for row in test_rows if row["method"] == method), key=lambda row: row["rep"])
        mse = np.array([row["test_mse"] for row in rows])
        pooled = np.array([row["pooled_psnr"] for row in rows])
        mean_image = np.array([row["mean_image_psnr"] for row in rows])
        mse_mean, mse_low, mse_high = mean_t_interval(mse)
        pooled_mean, pooled_low, pooled_high = mean_t_interval(pooled)
        image_mean, image_low, image_high = mean_t_interval(mean_image)
        summary_rows.append({
            "method": method,
            "selected_parameters": json.dumps([row["parameter"] for row in rows]),
            "test_mse": mse_mean,
            "test_mse_seed_t_low": mse_low,
            "test_mse_seed_t_high": mse_high,
            "pooled_psnr": pooled_mean,
            "pooled_psnr_seed_t_low": pooled_low,
            "pooled_psnr_seed_t_high": pooled_high,
            "mean_image_psnr": image_mean,
            "mean_image_psnr_seed_t_low": image_low,
            "mean_image_psnr_seed_t_high": image_high,
        })
    save_csv(out / "summary.csv", summary_rows)

    comparison_rows: list[dict] = []
    comparisons = [
        ("thinning_centered", "student_centered"),
        ("thinning_clean_oracle", "student_clean_oracle"),
        ("thinning_fixed_r05", "student_fixed_tau1"),
    ]
    for left, right in comparisons:
        left_rows = sorted((row for row in test_rows if row["method"] == left), key=lambda row: row["rep"])
        right_rows = sorted((row for row in test_rows if row["method"] == right), key=lambda row: row["rep"])
        for metric in ("test_mse", "pooled_psnr", "mean_image_psnr"):
            diff = np.array([a[metric] - b[metric] for a, b in zip(left_rows, right_rows)])
            mean, low, high = mean_t_interval(diff)
            comparison_rows.append({"left": left, "right": right, "metric": metric,
                                    "mean_difference": mean, "seed_t_low": low, "seed_t_high": high})
    save_csv(out / "paired_comparisons.csv", comparison_rows)
    (out / "complete.json").write_text(json.dumps({
        "candidate_fits": len(candidate_rows),
        "selection_rows": len(choice_rows),
        "test_rows": len(test_rows),
        "completed": True,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="output/experiments/allocation_main_20261008")
    parser.add_argument("--out", default="output/experiments/gp_thinning_comparison_20261010")
    parser.add_argument("--r-values", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--test-seed", type=int, default=20261010404)
    parser.add_argument("--test-n", type=int, default=2048)
    run(parser.parse_args())
