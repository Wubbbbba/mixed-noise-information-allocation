"""Match validation and deployment MC counts for the GR2R controlled study.

This script reuses all self-supervised and supervised checkpoints.  It first
recomputes validation scores with the exact deployment K for each noise,
writes every selection, and only then loads the untouched confirmation
images.  Historical checkpoint fields named ``alpha`` are GR2R allocation
proportions ``a`` in (0, 1), not Gaussian auxiliary-noise amplitudes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import t as student_t

from run_rigorous import (
    ALPHAS,
    MC_SAMPLES_BY_NOISE,
    NOISES,
    PAPER_MAIN,
    PAPER_SENS,
    DnCNN,
    make_pair,
    observe,
    seed_all,
    test_eval,
)


METHODS = (
    "centered_R_matched",
    "clean_oracle_self_matched",
    "paper_sensitivity",
    "paper_main",
    "raw_J_old",
    "supervised_oracle_matched",
)
TEST_OBSERVATION_BASE = 1_610_000
TEST_RECORRUPTION_BASE = 1_620_000


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_model(path: Path, device: torch.device) -> DnCNN:
    model = DnCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def validation_scores(model, x, y, kind, a, device, seed, samples):
    seed_all(seed)
    model.eval()
    n = len(x)
    prediction_sum = torch.zeros_like(x)
    cross_sum = torch.zeros(n, device=device)
    with torch.no_grad():
        for _ in range(samples):
            for lo in range(0, n, 8):
                hi = min(n, lo + 8)
                y1, y2 = make_pair(y[lo:hi], kind, a)
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction = model(y1)
                prediction = prediction.float().clamp(0, 1)
                prediction_sum[lo:hi] += prediction
                cross_sum[lo:hi] += (prediction * y2).mean((1, 2, 3))
    mean_prediction = prediction_sum / samples
    q = (mean_prediction.square()).mean((1, 2, 3)) - 2.0 * cross_sum / samples
    clean = ((mean_prediction - x).square()).mean((1, 2, 3))
    return q.cpu().double().numpy(), clean.cpu().double().numpy()


def load_benchmark(root: Path, crop: int = 128) -> torch.Tensor:
    files = sorted((root / "x2").glob("*.png"))
    if len(files) != 100:
        raise ValueError(f"expected 100 benchmark x2 images, found {len(files)}")
    images = []
    for path in files:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width, _ = array.shape
        if height < crop or width < crop:
            raise ValueError(f"image too small: {path} {array.shape}")
        top, left = (height - crop) // 2, (width - crop) // 2
        patch = array[top : top + crop, left : left + crop]
        images.append(torch.from_numpy(patch.copy()).permute(2, 0, 1))
    return torch.stack(images).float().div_(255)


def metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mse": float(values.mean()),
        "pooled_psnr_db": float(10.0 * np.log10(1.0 / values.mean())),
        "mean_per_image_psnr_db": float(np.mean(10.0 * np.log10(1.0 / np.maximum(values, 1e-12)))),
        "n_images": int(values.size),
    }


def rep_ci(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    mean = float(a.mean())
    half = float(student_t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a)))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half}


def main(args: argparse.Namespace) -> None:
    run = Path(args.run).resolve()
    supervised = Path(args.supervised_run).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "per_image").mkdir()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cache = torch.load(run / "data_cache.pt", map_location="cpu")
    x_val = cache["val"].float().div(255).to(device)

    candidate_rows, selection_rows = [], []
    selections: dict[tuple[int, str, str], tuple[float, str]] = {}
    # Validation and every selection are completed before the confirmation set is loaded.
    for seed in range(5):
        for noise_index, kind in enumerate(NOISES):
            deployment_k = int(MC_SAMPLES_BY_NOISE[kind])
            seed_all(100_000 * seed + 1_000 * noise_index + 12)
            y_val = observe(x_val, kind)
            self_local, supervised_local = [], []
            for a in ALPHAS:
                # Preserve each original experiment's validation auxiliary
                # stream.  The only intended self-supervised change is
                # Poisson K=10 -> K=5.
                self_score_seed = 300_000 * seed + 3_000 * noise_index + 41
                supervised_score_seed = 710_000 + 10_000 * seed + 1_000 * noise_index + 41
                self_model = load_model(run / f"seed{seed}" / "checkpoints" / f"{kind}_a{a:.2f}.pt", device)
                q, clean = validation_scores(self_model, x_val, y_val, kind, a, device, self_score_seed, deployment_k)
                # Keep ``deployment_j`` in serialized files for compatibility
                # with the completed experiment.  In the paper this count is K;
                # it is unrelated to the stability objective J_stab.
                self_row = {"noise": kind, "seed": seed, "training": "self_supervised", "a": a,
                            "deployment_j": deployment_k,
                            "val_q_mean": float(q.mean()), "val_clean_mse": float(clean.mean())}
                candidate_rows.append(self_row); self_local.append(self_row)
                del self_model

                supervised_model = load_model(supervised / f"seed{seed}" / "checkpoints" / f"{kind}_a{a:.2f}.pt", device)
                _unused, supervised_clean = validation_scores(supervised_model, x_val, y_val, kind, a, device, supervised_score_seed, deployment_k)
                supervised_row = {"noise": kind, "seed": seed, "training": "supervised", "a": a,
                                  "deployment_j": deployment_k,
                                  "val_q_mean": "", "val_clean_mse": float(supervised_clean.mean())}
                candidate_rows.append(supervised_row); supervised_local.append(supervised_row)
                del supervised_model
                torch.cuda.empty_cache()

            old = json.loads((run / f"seed{seed}" / f"{kind}_selection.json").read_text())
            selected = {
                "centered_R_matched": (min(self_local, key=lambda r: r["val_q_mean"])["a"], "self_supervised"),
                "clean_oracle_self_matched": (min(self_local, key=lambda r: r["val_clean_mse"])["a"], "self_supervised"),
                "paper_sensitivity": (PAPER_SENS[kind], "self_supervised"),
                "paper_main": (PAPER_MAIN[kind], "self_supervised"),
                "raw_J_old": (old["raw_J"], "self_supervised"),
                "supervised_oracle_matched": (min(supervised_local, key=lambda r: r["val_clean_mse"])["a"], "supervised"),
            }
            for method, (a, training) in selected.items():
                selections[(seed, kind, method)] = (float(a), training)
                # ``deployment_j`` is a legacy schema key denoting the MC count K.
                selection_rows.append({"noise": kind, "seed": seed, "method": method, "training": training,
                                       "a": a, "deployment_j": deployment_k})
            del y_val

    write_csv(out / "validation_candidates.csv", candidate_rows)
    write_csv(out / "selections_locked_before_test.csv", selection_rows)
    lock = {
        "selection_locked_before_test_load": True,
        "validation_j_equals_test_j": MC_SAMPLES_BY_NOISE,
        "selection_sha256": hashlib.sha256((out / "selections_locked_before_test.csv").read_bytes()).hexdigest(),
        "parameter_coordinate": "historical alpha field is GR2R allocation proportion a; Gaussian t=a/(1-a)",
        "old_checkpoints_reused": True,
        "new_training": False,
    }
    (out / "selection_lock.json").write_text(json.dumps(lock, indent=2))

    x_test_cpu = load_benchmark(Path(args.test_root))
    x_test = x_test_cpu.to(device)
    test_rows = []
    arrays: dict[tuple[str, str], list[np.ndarray]] = {(kind, method): [] for kind in NOISES for method in METHODS}
    for seed in range(5):
        for noise_index, kind in enumerate(NOISES):
            deployment_k = int(MC_SAMPLES_BY_NOISE[kind])
            seed_all(TEST_OBSERVATION_BASE + 10_000 * seed + 1_000 * noise_index + 51)
            y_test = observe(x_test, kind)
            for method in METHODS:
                a, training = selections[(seed, kind, method)]
                base = supervised if training == "supervised" else run / f"seed{seed}"
                checkpoint = (base / f"seed{seed}" / "checkpoints" / f"{kind}_a{a:.2f}.pt") if training == "supervised" else (base / "checkpoints" / f"{kind}_a{a:.2f}.pt")
                model = load_model(checkpoint, device)
                _single, mc = test_eval(model, x_test, y_test, kind, a, device,
                                        TEST_RECORRUPTION_BASE + 10_000 * seed + 1_000 * noise_index + 61,
                                        deployment_k)
                arrays[(kind, method)].append(mc)
                row = {"noise": kind, "seed": seed, "method": method, "training": training,
                       "a": a, "deployment_j": deployment_k, **metrics(mc)}
                test_rows.append(row)
                np.savez_compressed(out / "per_image" / f"{kind}_seed{seed}_{method}_a{a:.2f}.npz", mc_mse=mc)
                print(json.dumps({"event": "benchmark_test", **row}), flush=True)
                del model
                torch.cuda.empty_cache()
            del y_test

    write_csv(out / "test_by_seed.csv", test_rows)
    summary_rows = []
    for kind in NOISES:
        for method in METHODS:
            group = [r for r in test_rows if r["noise"] == kind and r["method"] == method]
            mse_ci = rep_ci([r["mse"] for r in group])
            pooled_rep_ci = rep_ci([r["pooled_psnr_db"] for r in group])
            image_rep_ci = rep_ci([r["mean_per_image_psnr_db"] for r in group])
            flat = np.concatenate(arrays[(kind, method)])
            summary_rows.append({
                "noise": kind, "method": method,
                "a_by_seed": ",".join(f"{r['a']:.2f}" for r in group),
                "mse": mse_ci["mean"], "mse_ci95_low": mse_ci["ci95_low"], "mse_ci95_high": mse_ci["ci95_high"],
                "pooled_psnr_db": float(10 * np.log10(1 / flat.mean())),
                "pooled_psnr_seed_ci95_low_db": pooled_rep_ci["ci95_low"], "pooled_psnr_seed_ci95_high_db": pooled_rep_ci["ci95_high"],
                "mean_per_image_psnr_db": float(np.mean(10 * np.log10(1 / np.maximum(flat, 1e-12)))),
                "mean_per_image_psnr_seed_ci95_low_db": image_rep_ci["ci95_low"], "mean_per_image_psnr_seed_ci95_high_db": image_rep_ci["ci95_high"],
            })
    write_csv(out / "test_summary.csv", summary_rows)

    paired_rows = []
    for kind in NOISES:
        target = {int(r["seed"]): r for r in test_rows if r["noise"] == kind and r["method"] == "centered_R_matched"}
        for baseline in ("paper_sensitivity", "paper_main", "clean_oracle_self_matched", "supervised_oracle_matched"):
            base = {int(r["seed"]): r for r in test_rows if r["noise"] == kind and r["method"] == baseline}
            mse_delta = [target[s]["mse"] - base[s]["mse"] for s in range(5)]
            psnr_delta = [target[s]["pooled_psnr_db"] - base[s]["pooled_psnr_db"] for s in range(5)]
            mci, pci = rep_ci(mse_delta), rep_ci(psnr_delta)
            paired_rows.append({"noise": kind, "comparison": f"centered_R_matched_minus_{baseline}", "mse_difference": mci["mean"], "mse_difference_ci95_low": mci["ci95_low"], "mse_difference_ci95_high": mci["ci95_high"], "pooled_psnr_difference_db": pci["mean"], "pooled_psnr_difference_ci95_low_db": pci["ci95_low"], "pooled_psnr_difference_ci95_high_db": pci["ci95_high"]})
    write_csv(out / "paired_differences.csv", paired_rows)
    dataset_name = Path(args.test_root).name
    protocol = {**lock, "source_run": str(run), "supervised_run": str(supervised), "test_dataset": f"{dataset_name} x2 HR, 100 center crops of 128x128", "test_root": str(Path(args.test_root).resolve()), "test_observation_seed_base": TEST_OBSERVATION_BASE, "test_recorruption_seed_base": TEST_RECORRUPTION_BASE, "validation_streams": "original self-supervised and supervised validation recorruption seeds preserved; only deployment K is matched", "test_loaded_only_after_selection_lock": True, "interval": f"two-sided Student t across five training seeds, df=4; fixed {dataset_name} clean images with independent observation/recorruption streams per seed"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (out / "complete.json").write_text(json.dumps({"complete": True, "candidate_rows": len(candidate_rows), "selection_rows": len(selection_rows), "test_rows": len(test_rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--supervised-run", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
