r"""Frozen five-seed GR2R allocation protocol used by the paper.

The candidate variable ``alpha`` is the GR2R allocation proportion
\(\alpha\in(0,1)\).  For the Gaussian construction it maps to the analytic
coordinates as ``t = alpha / (1 - alpha)``, ``rho = 1 - alpha``, and
``tau = sqrt(q * alpha / (1 - alpha))``.

Historical result files use ``raw_J``/``val_raw_J`` for the mean-plus-standard-
deviation stability objective and ``MC_J`` for the number of Monte Carlo
recorruptions.  These serialized names are retained for reproducibility; the
paper denotes the corresponding quantities by ``J_stab`` and ``K``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


GR2R_ALLOCATION_GRID = [0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
NOISES = ["gaussian", "poisson", "gamma"]
PAPER_SENS = {"gaussian": 0.30, "poisson": 0.12, "gamma": 0.12}
PAPER_MAIN = {"gaussian": 0.50, "poisson": 0.15, "gamma": 0.20}
MC_SAMPLES_BY_NOISE = {"gaussian": 10, "poisson": 5, "gamma": 10}
VALIDATION_MC_SAMPLES = 10

# Backward-compatible aliases imported by the original evaluation scripts.
ALPHAS = GR2R_ALLOCATION_GRID
MC_J = MC_SAMPLES_BY_NOISE
K_VAL = VALIDATION_MC_SAMPLES

TRAIN_IMAGES = 650
BATCH = 16
EPOCHS = 250
STEPS = math.ceil(TRAIN_IMAGES / BATCH) * EPOCHS


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DnCNN(nn.Module):
    def __init__(self, depth: int = 17, nf: int = 64):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, nf, 3, padding=1), nn.ReLU(True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                nn.BatchNorm2d(nf),
                nn.ReLU(True),
            ]
        layers += [nn.Conv2d(nf, 3, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.body(x)


def observe(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "gaussian":
        return x + 0.10 * torch.randn_like(x)
    if kind == "poisson":
        return 0.50 * torch.poisson(torch.clamp(x, min=0) / 0.50)
    ell = 5.0
    return torch._standard_gamma(torch.full_like(x, ell)) * torch.clamp(x, min=1e-6) / ell


def make_pair(y: torch.Tensor, kind: str, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the GR2R pair for allocation proportion ``alpha``."""
    if kind == "gaussian":
        omega = 0.10 * torch.randn_like(y)
        y1 = y + math.sqrt(alpha / (1 - alpha)) * omega
        y2 = y - math.sqrt((1 - alpha) / alpha) * omega
        return y1, y2
    if kind == "poisson":
        counts = torch.round(torch.clamp(y, min=0) / 0.50)
        split = torch.binomial(counts, torch.full_like(counts, alpha))
        return 0.50 * (counts - split) / (1 - alpha), 0.50 * split / alpha
    ell = 5.0
    first = torch._standard_gamma(torch.full_like(y, ell * alpha))
    second = torch._standard_gamma(torch.full_like(y, ell * (1 - alpha)))
    fraction = first / (first + second).clamp_min(1e-12)
    return y * (1 - fraction) / (1 - alpha), y * fraction / alpha


def train_model(y, kind, alpha, device, seed):
    seed_all(seed)
    model = DnCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 41)
    tail = []
    model.train()
    for step in range(STEPS):
        index = torch.randint(0, len(y), (BATCH,), device=device, generator=generator)
        y1, y2 = make_pair(y[index], kind, alpha)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.mse_loss(model(y1), y2)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step >= STEPS - 200:
            tail.append(float(loss.detach()))
    return model, float(np.mean(tail))


def val_eval(model, x, y, kind, alpha, device, seed, k=VALIDATION_MC_SAMPLES):
    seed_all(seed)
    model.eval()
    n = len(x)
    raw = torch.zeros(n, device=device)
    centered = torch.zeros(n, device=device)
    single = torch.zeros(n, device=device)
    prediction_sum = torch.zeros_like(x)
    with torch.no_grad():
        for _ in range(k):
            for lo in range(0, n, 8):
                hi = min(n, lo + 8)
                y1, y2 = make_pair(y[lo:hi], kind, alpha)
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction = model(y1)
                prediction = prediction.float().clamp(0, 1)
                dims = (1, 2, 3)
                raw[lo:hi] += ((prediction - y2) ** 2).mean(dims)
                centered[lo:hi] += (prediction**2 - 2 * prediction * y2).mean(dims)
                single[lo:hi] += ((prediction - x[lo:hi]) ** 2).mean(dims)
                prediction_sum[lo:hi] += prediction
    raw /= k
    centered /= k
    single /= k
    mc = ((prediction_sum / k - x) ** 2).mean((1, 2, 3))

    def as_array(tensor):
        return tensor.detach().cpu().double().numpy()

    raw_array, q_array, single_array, mc_array = map(as_array, (raw, centered, single, mc))
    mean_raw = raw_array.mean()
    sd_raw = raw_array.std(ddof=1)
    return {
        "val_q_mean": float(q_array.mean()),
        "val_q_sd_image": float(q_array.std(ddof=1)),
        "val_raw_mean": float(mean_raw),
        "val_raw_sd_image": float(sd_raw),
        "val_raw_J": float(mean_raw + sd_raw),  # legacy field: manuscript J_stab
        "val_single_mse": float(single_array.mean()),
        "val_mc_mse": float(mc_array.mean()),
        "val_single_pooled_psnr": float(10 * np.log10(1 / single_array.mean())),
        "val_mc_pooled_psnr": float(10 * np.log10(1 / mc_array.mean())),
    }


def test_eval(model, x, y, kind, alpha, device, seed, k):
    seed_all(seed)
    model.eval()
    n = len(x)
    single = torch.zeros(n, device=device)
    prediction_sum = torch.zeros_like(x)
    with torch.no_grad():
        for _ in range(k):
            for lo in range(0, n, 8):
                hi = min(n, lo + 8)
                y1, _ = make_pair(y[lo:hi], kind, alpha)
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction = model(y1)
                prediction = prediction.float().clamp(0, 1)
                single[lo:hi] += ((prediction - x[lo:hi]) ** 2).mean((1, 2, 3))
                prediction_sum[lo:hi] += prediction
    single /= k
    mc = ((prediction_sum / k - x) ** 2).mean((1, 2, 3))
    return single.cpu().double().numpy(), mc.cpu().double().numpy()


def metric(values):
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(n)
    return {
        "mse": mean,
        "ci_low": max(0.0, mean - 1.96 * se),
        "ci_high": mean + 1.96 * se,
        "pooled_psnr": float(10 * np.log10(1 / mean)),
        "mean_image_psnr": float(np.mean(10 * np.log10(1 / np.maximum(values, 1e-12)))),
        "n_images": n,
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "per_image").mkdir(exist_ok=True)
    cache = torch.load("data_cache.pt", map_location="cpu")
    x_train = cache["train"].float().div_(255).to(device)
    x_val = cache["val"].float().div_(255).to(device)
    protocol = {
        "seed": args.seed,
        "alphas": ALPHAS,
        "parameter_coordinate": "alpha is the GR2R allocation proportion; Gaussian t=alpha/(1-alpha)",
        "noise": {"gaussian_sigma": 0.1, "poisson_gain": 0.5, "gamma_looks": 5},
        "architecture": "standard residual DnCNN depth=17 width=64 with BatchNorm",
        "optimizer": "Adam lr=1e-4 constant",
        "amp": True,
        "batch_size": BATCH,
        "epochs_image_equivalent": EPOCHS,
        "steps": STEPS,
        "train_images": 650,
        "val_images": 75,
        "test_images": 75,
        "train_crop": 64,
        "eval_center_crop": 128,
        "val_recorruptions": K_VAL,
        "test_mc_samples": MC_J,
        "notation": "legacy raw_J means J_stab; legacy MC_J means K by noise family",
        "selection": "validation mean centered Q only",
        "test_policy": "not evaluated until selection.json is written",
        "paper_sensitivity_alpha": PAPER_SENS,
        "paper_main_alpha": PAPER_MAIN,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))

    all_candidates, all_tests = [], []
    for noise_index, kind in enumerate(NOISES):
        seed_all(100000 * args.seed + 1000 * noise_index + 11)
        y_train = observe(x_train, kind)
        seed_all(100000 * args.seed + 1000 * noise_index + 12)
        y_val = observe(x_val, kind)
        local = []
        for alpha in ALPHAS:
            tag = f"{kind}_a{alpha:.2f}"
            checkpoint = out / "checkpoints" / f"{tag}.pt"
            record = out / f"{tag}.json"
            if checkpoint.exists() and record.exists():
                row = json.loads(record.read_text())
                local.append(row)
                all_candidates.append(row)
                print(json.dumps({"event": "resume", **row}), flush=True)
                continue
            started = time.time()
            model, tail_loss = train_model(
                y_train, kind, alpha, device, 200000 * args.seed + 2000 * noise_index + 31
            )
            validation = val_eval(
                model, x_val, y_val, kind, alpha, device,
                300000 * args.seed + 3000 * noise_index + 41, K_VAL,
            )
            row = {
                "noise": kind,
                "seed": args.seed,
                "alpha": alpha,
                "train_tail_loss": tail_loss,
                "seconds": time.time() - started,
                **validation,
            }
            torch.save(model.state_dict(), checkpoint)
            record.write_text(json.dumps(row, indent=2))
            local.append(row)
            all_candidates.append(row)
            write_csv(out / "candidates.partial.csv", all_candidates)
            print(json.dumps({"event": "candidate", **row}), flush=True)
            del model
            torch.cuda.empty_cache()

        selected = {
            "centered_R": min(local, key=lambda row: row["val_q_mean"])["alpha"],
            "clean_oracle_single": min(local, key=lambda row: row["val_single_mse"])["alpha"],
            "clean_oracle_mc": min(local, key=lambda row: row["val_mc_mse"])["alpha"],
            "raw_J": min(local, key=lambda row: row["val_raw_J"])["alpha"],
            "paper_sensitivity": PAPER_SENS[kind],
            "paper_main": PAPER_MAIN[kind],
        }
        (out / f"{kind}_selection.json").write_text(json.dumps(selected, indent=2))

        # Test data are first accessed after validation-only selection is frozen.
        x_test = cache["test"].float().div(255).to(device)
        seed_all(400000 * args.seed + 4000 * noise_index + 51)
        y_test = observe(x_test, kind)
        for method, alpha in selected.items():
            model = DnCNN().to(device)
            model.load_state_dict(
                torch.load(out / "checkpoints" / f"{kind}_a{alpha:.2f}.pt", map_location=device)
            )
            single, mc = test_eval(
                model, x_test, y_test, kind, alpha, device,
                500000 * args.seed + 5000 * noise_index + 61, MC_J[kind],
            )
            np.savez_compressed(
                out / "per_image" / f"{kind}_{method}.npz", single_mse=single, mc_mse=mc
            )
            validation = next(row for row in local if row["alpha"] == alpha)
            row = {
                "noise": kind,
                "seed": args.seed,
                "method": method,
                "alpha": alpha,
                "val_q_mean": validation["val_q_mean"],
                "val_single_mse": validation["val_single_mse"],
                "val_mc_mse": validation["val_mc_mse"],
                **{f"test_single_{key}": value for key, value in metric(single).items()},
                **{f"test_mc_{key}": value for key, value in metric(mc).items()},
            }
            all_tests.append(row)
            print(json.dumps({"event": "test", **row}), flush=True)
            del model
            torch.cuda.empty_cache()
        del y_train, y_val, y_test, x_test
        torch.cuda.empty_cache()

    write_csv(out / "candidates.csv", all_candidates)
    write_csv(out / "test_results.csv", all_tests)
    (out / "complete.json").write_text(
        json.dumps({"complete": True, "seed": args.seed, "time": time.time()}, indent=2)
    )


if __name__ == "__main__":
    main()
