"""Train an input-matched clean-label reference for the GR2R alpha study.

For every seed, noise family, and candidate alpha, this script uses the same
DnCNN, fixed noisy training observations, recorruption kernel, and training
budget as ``run_rigorous.py``.  The only change is the target:

    self-supervised:  ||f_alpha(Y1_alpha) - Y2_alpha||^2
    supervised:       ||f_alpha(Y1_alpha) - X||^2

The clean validation set selects the supervised candidate before a fresh test
observation is generated.  The result is an empirical, input-matched
supervised reference within the finite alpha grid; it is not a mathematical
upper bound over architectures or optimization procedures.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_rigorous import (
    ALPHAS,
    BATCH,
    K_VAL,
    MC_J,
    NOISES,
    STEPS,
    DnCNN,
    make_pair,
    metric,
    observe,
    seed_all,
    test_eval,
)

TRAIN_SEED_BASE = 700_000
VALIDATION_RECORRUPTION_BASE = 710_000
TEST_OBSERVATION_BASE = 910_000
TEST_RECORRUPTION_BASE = 920_000


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_supervised(x, y, kind, alpha, device, seed):
    """Fit f(Y1_alpha) to clean X with the original optimization budget."""
    seed_all(seed)
    model = DnCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 41)
    tail: list[float] = []
    model.train()
    for step in range(STEPS):
        index = torch.randint(0, len(y), (BATCH,), device=device, generator=generator)
        clean = x[index]
        noisy = y[index]
        y1, _ = make_pair(noisy, kind, alpha)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.mse_loss(model(y1), clean)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step >= STEPS - 200:
            tail.append(float(loss.detach()))
    return model, float(np.mean(tail))


def validation_eval(model, x, y, kind, alpha, device, seed, samples=K_VAL):
    """Return per-image clean risks for single and finite-K MC inference."""
    seed_all(seed)
    model.eval()
    n = len(x)
    single_sum = torch.zeros(n, device=device)
    prediction_sum = torch.zeros_like(x)
    with torch.no_grad():
        for _ in range(samples):
            for lo in range(0, n, 8):
                hi = min(n, lo + 8)
                y1, _ = make_pair(y[lo:hi], kind, alpha)
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction = model(y1)
                prediction = prediction.float().clamp(0, 1)
                single_sum[lo:hi] += ((prediction - x[lo:hi]) ** 2).mean((1, 2, 3))
                prediction_sum[lo:hi] += prediction
    single = single_sum / samples
    mean_prediction = prediction_sum / samples
    mc = ((mean_prediction - x) ** 2).mean((1, 2, 3))
    return single.cpu().double().numpy(), mc.cpu().double().numpy()


def candidate_record(noise, seed, alpha, tail_loss, seconds, single, mc):
    return {
        "noise": noise,
        "seed": seed,
        "alpha": alpha,
        "train_tail_supervised_loss": tail_loss,
        "seconds": seconds,
        "val_single_mse": float(single.mean()),
        "val_single_pooled_psnr": float(10 * np.log10(1 / single.mean())),
        "val_mc_mse": float(mc.mean()),
        "val_mc_pooled_psnr": float(10 * np.log10(1 / mc.mean())),
        "val_images": len(single),
        "val_recorruptions": K_VAL,
    }


def evaluate_seed(run: Path, output_root: Path, seed: int, device: torch.device) -> None:
    source = run / f"seed{seed}"
    if not (source / "complete.json").exists():
        raise RuntimeError(f"source seed {seed} is not complete")
    output = output_root / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    per_image = output / "per_image"
    per_image.mkdir(exist_ok=True)

    cache = torch.load(run / "data_cache.pt", map_location="cpu")
    x_train = cache["train"].float().div_(255).to(device)
    x_val = cache["val"].float().div_(255).to(device)
    protocol = {
        "seed": seed,
        "source_run": str(run),
        "alphas": ALPHAS,
        "noise_families": NOISES,
        "architecture": "standard residual DnCNN depth=17 width=64 with BatchNorm",
        "optimizer": "Adam lr=1e-4 constant",
        "amp": True,
        "batch_size": BATCH,
        "steps": STEPS,
        "train_images": len(x_train),
        "val_images": len(x_val),
        "test_images": int(len(cache["test"])),
        "supervised_target": "clean X",
        "model_input": "Y1_alpha generated from the same fixed Y used by self-supervised training",
        "selection": "clean validation MSE within the finite alpha grid",
        "test_policy": "test tensor and fresh observation are accessed only after selection.json is written",
        "validation_recorruptions": K_VAL,
        "test_mc_samples": MC_J,
        "fresh_test_observation_base": TEST_OBSERVATION_BASE,
        "fresh_test_recorruption_base": TEST_RECORRUPTION_BASE,
        "interpretation": "empirical input-matched supervised reference, not a mathematical upper bound",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    candidate_rows: list[dict] = []
    test_rows: list[dict] = []
    for noise_index, kind in enumerate(NOISES):
        seed_all(100_000 * seed + 1_000 * noise_index + 11)
        y_train = observe(x_train, kind)
        seed_all(100_000 * seed + 1_000 * noise_index + 12)
        y_val = observe(x_val, kind)
        local: list[dict] = []
        for alpha in ALPHAS:
            tag = f"{kind}_a{alpha:.2f}"
            checkpoint = checkpoints / f"{tag}.pt"
            record_path = output / f"{tag}.json"
            if checkpoint.exists() and record_path.exists():
                row = json.loads(record_path.read_text())
                local.append(row)
                candidate_rows.append(row)
                print(json.dumps({"event": "resume", **row}), flush=True)
                continue

            started = time.time()
            model, tail_loss = train_supervised(
                x_train,
                y_train,
                kind,
                alpha,
                device,
                TRAIN_SEED_BASE + 10_000 * seed + 1_000 * noise_index + 31,
            )
            single, mc = validation_eval(
                model,
                x_val,
                y_val,
                kind,
                alpha,
                device,
                VALIDATION_RECORRUPTION_BASE + 10_000 * seed + 1_000 * noise_index + 41,
                K_VAL,
            )
            row = candidate_record(
                kind, seed, alpha, tail_loss, time.time() - started, single, mc
            )
            torch.save(model.state_dict(), checkpoint)
            record_path.write_text(json.dumps(row, indent=2))
            local.append(row)
            candidate_rows.append(row)
            write_csv(output / "candidates.partial.csv", candidate_rows)
            print(json.dumps({"event": "candidate", **row}), flush=True)
            del model
            torch.cuda.empty_cache()

        selected = {
            "supervised_oracle_single": min(local, key=lambda row: row["val_single_mse"])["alpha"],
            "supervised_oracle_mc": min(local, key=lambda row: row["val_mc_mse"])["alpha"],
        }
        selection_record = {
            "noise": kind,
            "seed": seed,
            "selection_locked_before_fresh_test": True,
            "selections": selected,
        }
        (output / f"{kind}_selection.json").write_text(
            json.dumps(selection_record, indent=2)
        )

        # Only now load/access the test tensor and draw its fresh observation.
        x_test = cache["test"].float().div(255).to(device)
        seed_all(TEST_OBSERVATION_BASE + 10_000 * seed + 1_000 * noise_index + 51)
        y_test = observe(x_test, kind)
        evaluated: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for alpha in sorted(set(float(value) for value in selected.values())):
            model = DnCNN().to(device)
            model.load_state_dict(
                torch.load(checkpoints / f"{kind}_a{alpha:.2f}.pt", map_location=device)
            )
            single, mc = test_eval(
                model,
                x_test,
                y_test,
                kind,
                alpha,
                device,
                TEST_RECORRUPTION_BASE + 10_000 * seed + 1_000 * noise_index + 61,
                MC_J[kind],
            )
            evaluated[alpha] = (single, mc)
            np.savez_compressed(
                per_image / f"{kind}_a{alpha:.2f}.npz",
                single_mse=single,
                mc_mse=mc,
            )
            del model
            torch.cuda.empty_cache()

        for method, alpha_value in selected.items():
            alpha = float(alpha_value)
            single, mc = evaluated[alpha]
            val_row = next(row for row in local if float(row["alpha"]) == alpha)
            row = {
                "noise": kind,
                "seed": seed,
                "method": method,
                "alpha": alpha,
                "val_single_mse": val_row["val_single_mse"],
                "val_mc_mse": val_row["val_mc_mse"],
                **{f"fresh_test_single_{key}": value for key, value in metric(single).items()},
                **{f"fresh_test_mc_{key}": value for key, value in metric(mc).items()},
            }
            test_rows.append(row)
            print(json.dumps({"event": "test", **row}), flush=True)

        del y_train, y_val, y_test, x_test
        torch.cuda.empty_cache()

    write_csv(output / "candidates.csv", candidate_rows)
    write_csv(output / "test_results.csv", test_rows)
    (output / "complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "seed": seed,
                "training_performed": True,
                "selection_uses_validation_only": True,
                "test_evaluated_after_selection_lock": True,
                "completed_at_unix": time.time(),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    for seed in args.seeds:
        evaluate_seed(run, output, seed, device)


if __name__ == "__main__":
    main()
