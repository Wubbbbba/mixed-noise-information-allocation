"""Label-free convex fusion of the five frozen mixed-noise CNN repetitions.

The protocol uses a new common validation stream, independent of every original
training/validation split.  For each noise law and each common allocation tau,
the script evaluates the five frozen networks on exactly the same (Z_tau,T_tau)
pairs, then fits: (i) a global simplex weight vector and (ii) four simplex
vectors on a prespecified 2x2 spatial partition.  Allocation and weights are
written to a lock file before the independent test stream is generated.

No network is retrained.  Clean validation risks are saved only as diagnostics;
they do not enter any reported label-free selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import t as student_t
import torch

from noise_models import make_pair, sample_observation
from run_allocation import BoundedCNN, HIGH, LOW, sample_images


MODELS = {"pg": 1, "gg": 2, "gp": 3}
METHODS = ("best_single_Q", "uniform", "global_Q", "regional_Q")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def quadrature_order(base_order: int, tau: float) -> int:
    return max(base_order, 48) if tau <= 0.25 else base_order


def load_models(checkpoint_dir: Path, noise: str, tau: float, reps: int, device: torch.device):
    models = []
    paths = []
    for rep in range(reps):
        path = checkpoint_dir / f"{noise}_r{rep}_tau{tau:g}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        model = BoundedCNN().to(device)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        paths.append(path)
    return models, paths


def predict_many(models, z: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for model in models:
            chunks = []
            for lo in range(0, len(z), batch_size):
                x = torch.from_numpy(np.ascontiguousarray(z[lo:lo + batch_size], dtype=np.float32)).to(device)
                chunks.append(model(x).cpu().numpy())
            outputs.append(np.concatenate(chunks).astype(np.float64, copy=False))
    return np.stack(outputs, axis=0)  # S x N x 1 x H x W


def solve_simplex_qp(g: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Solve min w'Gw-2c'w on the simplex, with multi-start verification."""
    s = len(c)
    g = 0.5 * (g + g.T)

    def fun(w):
        return float(w @ g @ w - 2.0 * c @ w)

    def jac(w):
        return 2.0 * (g @ w - c)

    starts = [np.full(s, 1.0 / s)] + [np.eye(s)[j] for j in range(s)]
    candidates = []
    for start in starts:
        result = minimize(
            fun,
            start,
            jac=jac,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * s,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                         "jac": lambda w: np.ones_like(w)},
            options={"ftol": 1e-13, "maxiter": 2000, "disp": False},
        )
        if result.success and np.all(np.isfinite(result.x)):
            w = np.maximum(result.x, 0.0)
            w /= w.sum()
            candidates.append((fun(w), w))
    if not candidates:
        raise RuntimeError("simplex quadratic program failed for every start")
    value, weight = min(candidates, key=lambda item: item[0])
    feasibility = max(abs(weight.sum() - 1.0), float(np.maximum(-weight, 0.0).max()))
    return weight, value, feasibility


def gram_cross(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None):
    # pred: S x N x 1 x H x W.  Every score retains the full-image 1/(N d) normalization.
    s, n, _, h, w = pred.shape
    if mask is None:
        flat = pred.reshape(s, -1)
        tflat = target.reshape(-1)
    else:
        flat = pred[:, :, :, mask].reshape(s, -1)
        tflat = target[:, :, mask].reshape(-1)
    scale = float(n * h * w)
    return (flat @ flat.T) / scale, (flat @ tflat) / scale


def score_prediction(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(prediction**2 - 2.0 * prediction * target))


def clean_per_image(prediction: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return np.mean((prediction - clean) ** 2, axis=(1, 2, 3))


def global_fuse(pred: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, pred, axes=(0, 0))


def region_masks(size: int) -> list[np.ndarray]:
    if size % 2:
        raise ValueError("the fixed 2x2 partition requires an even image size")
    half = size // 2
    masks = []
    for row in range(2):
        for col in range(2):
            mask = np.zeros((size, size), dtype=bool)
            mask[row * half:(row + 1) * half, col * half:(col + 1) * half] = True
            masks.append(mask)
    return masks


def regional_fuse(pred: np.ndarray, weights: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(pred[0])
    for p, mask in enumerate(masks):
        fused = global_fuse(pred, weights[p])
        out[:, :, mask] = fused[:, :, mask]
    return out


def summarize_mse(values: np.ndarray) -> dict:
    n = len(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    radius = float(student_t.ppf(0.975, n - 1) * se)
    dynamic_range = HIGH - LOW
    return {
        "mse": mean,
        "mse_se": se,
        "mse_ci95_low": mean - radius,
        "mse_ci95_high": mean + radius,
        "pooled_psnr_db": float(10.0 * np.log10(dynamic_range**2 / mean)),
    }


def paired_summary(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    n = len(diff)
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n))
    radius = float(student_t.ppf(0.975, n - 1) * se)
    psnr_a = 10.0 * np.log10((HIGH - LOW) ** 2 / float(a.mean()))
    psnr_b = 10.0 * np.log10((HIGH - LOW) ** 2 / float(b.mean()))
    return {
        "mse_difference": mean,
        "mse_difference_se": se,
        "mse_difference_ci95_low": mean - radius,
        "mse_difference_ci95_high": mean + radius,
        "pooled_psnr_difference_db": float(psnr_a - psnr_b),
    }


def method_prediction(method: str, pred: np.ndarray, record: dict, masks: list[np.ndarray]) -> np.ndarray:
    if method == "best_single_Q":
        return pred[int(record["seed_index"])]
    if method == "uniform":
        return pred.mean(axis=0)
    if method == "global_Q":
        return global_fuse(pred, np.asarray(record["weights"], dtype=np.float64))
    if method == "regional_Q":
        return regional_fuse(pred, np.asarray(record["weights"], dtype=np.float64), masks)
    raise KeyError(method)


def run(args) -> None:
    started = time.perf_counter()
    source_run = Path(args.run).resolve()
    checkpoint_dir = source_run / "checkpoints"
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          args.device if args.device != "auto" else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    source_protocol = json.loads((source_run / "protocol.json").read_text(encoding="utf-8"))
    taus = [float(v) for v in source_protocol["taus"]]
    noises = list(source_protocol["models"])
    reps = int(source_protocol["reps"])
    size = int(source_protocol["size"])
    params = {key: float(source_protocol[key]) for key in ("gamma", "sigma", "ell")}
    quad_order = int(source_protocol["quad_order"])
    masks = region_masks(size)
    seeds = {
        "validation_clean": args.seed + 11,
        "validation_observation_base": args.seed + 1000,
        "validation_pair_base": args.seed + 2000,
        "test_clean": args.seed + 33,
        "test_observation_base": args.seed + 3000,
        "test_pair_base": args.seed + 4000,
    }
    protocol = {
        "source_run": str(source_run),
        "source_protocol_sha256": sha256(source_run / "protocol.json"),
        "checkpoint_dir": str(checkpoint_dir),
        "noise_models": noises,
        "taus": taus,
        "seed_repetitions": reps,
        "validation_n": args.val_n,
        "test_n": args.test_n,
        "image_size": size,
        "partition": "fixed 2x2 quadrants chosen before validation",
        "methods": list(METHODS),
        "selection": "centered Q only; clean validation is diagnostic",
        "test_generated_after_lock": True,
        "seeds": seeds,
        "device": str(device),
        "batch_size": args.batch_size,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "source_sha256": sha256(Path(__file__)),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    x_val = sample_images(args.val_n, size, np.random.default_rng(seeds["validation_clean"]))
    validation_rows = []
    lock: dict[str, dict] = {}
    checkpoint_hashes = {}

    for noise in noises:
        model_id = MODELS[noise]
        y_val = sample_observation(
            x_val,
            noise,
            np.random.default_rng(seeds["validation_observation_base"] + model_id),
            **params,
        )
        per_tau = []
        for tau_index, tau in enumerate(taus):
            z_val, t_val = make_pair(
                y_val,
                noise,
                tau,
                np.random.default_rng(seeds["validation_pair_base"] + 100 * model_id + tau_index),
                quad_order=quadrature_order(quad_order, tau),
                **params,
            )
            models, paths = load_models(checkpoint_dir, noise, tau, reps, device)
            for path in paths:
                checkpoint_hashes[path.name] = sha256(path)
            pred = predict_many(models, z_val, device, args.batch_size)
            single_q = np.asarray([score_prediction(pred[s], t_val) for s in range(reps)])
            single_clean = np.asarray([clean_per_image(pred[s], x_val).mean() for s in range(reps)])
            best_seed = int(np.argmin(single_q))
            uniform_w = np.full(reps, 1.0 / reps)
            uniform_pred = global_fuse(pred, uniform_w)
            g, c = gram_cross(pred, t_val)
            global_w, global_q, global_feas = solve_simplex_qp(g, c)
            global_pred = global_fuse(pred, global_w)
            regional_weights = []
            regional_q = 0.0
            regional_feas = 0.0
            for mask in masks:
                gp, cp = gram_cross(pred, t_val, mask)
                wp, qp, fp = solve_simplex_qp(gp, cp)
                regional_weights.append(wp)
                regional_q += qp
                regional_feas = max(regional_feas, fp)
            regional_weights = np.stack(regional_weights)
            regional_pred = regional_fuse(pred, regional_weights, masks)
            row = {
                "noise": noise,
                "tau": tau,
                "best_single_seed": best_seed,
                "best_single_val_Q": float(single_q[best_seed]),
                "best_single_clean_mse_diagnostic": float(single_clean[best_seed]),
                "uniform_val_Q": score_prediction(uniform_pred, t_val),
                "uniform_clean_mse_diagnostic": float(clean_per_image(uniform_pred, x_val).mean()),
                "global_val_Q": global_q,
                "global_clean_mse_diagnostic": float(clean_per_image(global_pred, x_val).mean()),
                "global_solver_feasibility": global_feas,
                "regional_val_Q": regional_q,
                "regional_clean_mse_diagnostic": float(clean_per_image(regional_pred, x_val).mean()),
                "regional_solver_feasibility": regional_feas,
                "global_weights": json.dumps(global_w.tolist()),
                "regional_weights": json.dumps(regional_weights.tolist()),
            }
            validation_rows.append(row)
            per_tau.append((row, global_w, regional_weights))

        best_single = min(per_tau, key=lambda item: item[0]["best_single_val_Q"])
        best_uniform = min(per_tau, key=lambda item: item[0]["uniform_val_Q"])
        best_global = min(per_tau, key=lambda item: item[0]["global_val_Q"])
        best_regional = min(per_tau, key=lambda item: item[0]["regional_val_Q"])
        lock[noise] = {
            "best_single_Q": {
                "tau": best_single[0]["tau"],
                "seed_index": best_single[0]["best_single_seed"],
                "validation_Q": best_single[0]["best_single_val_Q"],
            },
            "uniform": {
                "tau": best_uniform[0]["tau"],
                "weights": np.full(reps, 1.0 / reps).tolist(),
                "validation_Q": best_uniform[0]["uniform_val_Q"],
            },
            "global_Q": {
                "tau": best_global[0]["tau"],
                "weights": best_global[1].tolist(),
                "validation_Q": best_global[0]["global_val_Q"],
            },
            "regional_Q": {
                "tau": best_regional[0]["tau"],
                "weights": best_regional[2].tolist(),
                "validation_Q": best_regional[0]["regional_val_Q"],
            },
        }

    save_csv(out / "validation_candidates.csv", validation_rows)
    lock_document = {
        "locked_before_test_generation": True,
        "selection_uses_clean_validation": False,
        "selection": lock,
        "checkpoint_sha256": checkpoint_hashes,
    }
    lock_path = out / "locked_selection.json"
    lock_path.write_text(json.dumps(lock_document, indent=2), encoding="utf-8")
    lock_hash_before_test = sha256(lock_path)

    # Only after the lock exists do we instantiate either test clean images or observations.
    x_test = sample_images(args.test_n, size, np.random.default_rng(seeds["test_clean"]))
    test_arrays = {}
    summary_rows = []
    paired_rows = []
    for noise in noises:
        model_id = MODELS[noise]
        y_test = sample_observation(
            x_test,
            noise,
            np.random.default_rng(seeds["test_observation_base"] + model_id),
            **params,
        )
        prediction_cache = {}
        target_cache = {}
        for tau_index, tau in enumerate(taus):
            if tau not in {float(lock[noise][method]["tau"]) for method in METHODS}:
                continue
            z_test, t_test = make_pair(
                y_test,
                noise,
                tau,
                np.random.default_rng(seeds["test_pair_base"] + 100 * model_id + tau_index),
                quad_order=quadrature_order(quad_order, tau),
                **params,
            )
            models, _ = load_models(checkpoint_dir, noise, tau, reps, device)
            prediction_cache[tau] = predict_many(models, z_test, device, args.batch_size)
            target_cache[tau] = t_test

        method_mse = {}
        for method in METHODS:
            record = lock[noise][method]
            tau = float(record["tau"])
            fused = method_prediction(method, prediction_cache[tau], record, masks)
            mse = clean_per_image(fused, x_test)
            method_mse[method] = mse
            test_arrays[f"{noise}_{method}_mse"] = mse
            row = {"noise": noise, "method": method, "tau": tau, "test_n": args.test_n,
                   "test_centered_Q": score_prediction(fused, target_cache[tau])}
            row.update(summarize_mse(mse))
            summary_rows.append(row)
        for method in ("global_Q", "regional_Q"):
            for baseline in ("best_single_Q", "uniform"):
                row = {"noise": noise, "method": method, "baseline": baseline,
                       "difference_definition": "method minus baseline"}
                row.update(paired_summary(method_mse[method], method_mse[baseline]))
                paired_rows.append(row)

    if sha256(lock_path) != lock_hash_before_test:
        raise AssertionError("selection lock changed after test generation")
    np.savez_compressed(out / "test_per_image.npz", **test_arrays)
    save_csv(out / "test_summary.csv", summary_rows)
    save_csv(out / "paired_differences.csv", paired_rows)
    complete = {
        "complete": True,
        "elapsed_seconds": time.perf_counter() - started,
        "locked_selection_sha256": lock_hash_before_test,
        "validation_rows": len(validation_rows),
        "test_rows": len(summary_rows),
        "paired_rows": len(paired_rows),
        "no_retraining": True,
        "test_set_used_for_selection": False,
    }
    (out / "complete.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="output/experiments/allocation_main_20261008")
    parser.add_argument("--out", default="output/experiments/seed_fusion_20261015")
    parser.add_argument("--val-n", type=int, default=1024)
    parser.add_argument("--test-n", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2026101500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="auto")
    run(parser.parse_args())
