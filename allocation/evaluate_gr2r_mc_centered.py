"""Evaluate a centered validation score for finite-K GR2R MC inference.

This script reuses frozen checkpoints.  It does not train or tune a model on
test data.  For f_j = f(Y1_j) and fbar = mean_j f_j, it evaluates

  Q_MC,K = (||fbar||^2 - 2/K sum_j <f_j, Y2_j>) / d,

whose expectation equals the finite-K MC reconstruction MSE minus E||X||^2/d.
Validation-only selections are serialized before a fresh test corruption is
drawn.  The clean validation oracle is diagnostic only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from run_rigorous import (
    ALPHAS,
    K_VAL,
    MC_J,
    NOISES,
    DnCNN,
    make_pair,
    metric,
    observe,
    seed_all,
    test_eval,
)

TEST_OBSERVATION_BASE = 910_000
TEST_RECORRUPTION_BASE = 920_000


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def val_mc_score(model, x, y, kind, alpha, device, seed, samples=K_VAL):
    seed_all(seed)
    model.eval()
    n = len(x)
    prediction_sum = torch.zeros_like(x)
    paired_cross_sum = torch.zeros(n, device=device)
    with torch.no_grad():
        for _ in range(samples):
            for lo in range(0, n, 8):
                hi = min(n, lo + 8)
                y1, y2 = make_pair(y[lo:hi], kind, alpha)
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction = model(y1)
                prediction = prediction.float().clamp(0, 1)
                prediction_sum[lo:hi] += prediction
                paired_cross_sum[lo:hi] += (prediction * y2).mean((1, 2, 3))
    mean_prediction = prediction_sum / samples
    q_mc = (mean_prediction * mean_prediction).mean((1, 2, 3))
    q_mc -= 2.0 * paired_cross_sum / samples
    clean_mc = ((mean_prediction - x) ** 2).mean((1, 2, 3))
    return q_mc.cpu().double().numpy(), clean_mc.cpu().double().numpy()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def evaluate_seed(run: Path, output_root: Path, seed: int, device: torch.device) -> None:
    source = run / f"seed{seed}"
    if not (source / "complete.json").exists():
        raise RuntimeError(f"seed {seed} is not complete")
    output = output_root / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_image").mkdir(exist_ok=True)

    cache = torch.load(run / "data_cache.pt", map_location="cpu")
    x_val = cache["val"].float().div(255).to(device)
    x_test = cache["test"].float().div(255).to(device)
    protocol = load_json(source / "protocol.json")
    candidate_rows: list[dict] = []
    test_rows: list[dict] = []

    for noise_index, kind in enumerate(NOISES):
        seed_all(100_000 * seed + 1_000 * noise_index + 12)
        y_val = observe(x_val, kind)
        local: list[dict] = []
        for alpha in ALPHAS:
            model = DnCNN().to(device)
            checkpoint = source / "checkpoints" / f"{kind}_a{alpha:.2f}.pt"
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            q_values, clean_values = val_mc_score(
                model,
                x_val,
                y_val,
                kind,
                alpha,
                device,
                300_000 * seed + 3_000 * noise_index + 41,
                K_VAL,
            )
            row = {
                "noise": kind,
                "seed": seed,
                "alpha": alpha,
                "mc_samples": K_VAL,
                "val_q_mc_mean": float(q_values.mean()),
                "val_q_mc_sd_image": float(q_values.std(ddof=1)),
                "val_clean_mc_mse": float(clean_values.mean()),
                "val_clean_mc_pooled_psnr": float(10 * np.log10(1 / clean_values.mean())),
            }
            local.append(row)
            candidate_rows.append(row)
            print(json.dumps({"event": "mc_candidate", **row}), flush=True)
            del model
            torch.cuda.empty_cache()

        old_selection = load_json(source / f"{kind}_selection.json")
        selected = {
            "centered_R_mc": min(local, key=lambda row: row["val_q_mc_mean"])["alpha"],
            "clean_oracle_mc": min(local, key=lambda row: row["val_clean_mc_mse"])["alpha"],
            "centered_R_single": old_selection["centered_R"],
            "clean_oracle_single": old_selection["clean_oracle_single"],
            "paper_sensitivity": old_selection["paper_sensitivity"],
            "paper_main": old_selection["paper_main"],
            "raw_J": old_selection["raw_J"],
        }
        selection_record = {
            "noise": kind,
            "seed": seed,
            "selection_locked_before_fresh_test": True,
            "validation_mc_samples": K_VAL,
            "selections": selected,
        }
        (output / f"{kind}_selection.json").write_text(
            json.dumps(selection_record, indent=2)
        )

        seed_all(TEST_OBSERVATION_BASE + 10_000 * seed + 1_000 * noise_index + 51)
        y_test = observe(x_test, kind)
        evaluated: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for alpha in sorted(set(float(value) for value in selected.values())):
            model = DnCNN().to(device)
            model.load_state_dict(
                torch.load(source / "checkpoints" / f"{kind}_a{alpha:.2f}.pt", map_location=device)
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
                output / "per_image" / f"{kind}_a{alpha:.2f}.npz",
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
                "val_q_mc_mean": val_row["val_q_mc_mean"],
                "val_clean_mc_mse": val_row["val_clean_mc_mse"],
                **{f"fresh_test_single_{key}": value for key, value in metric(single).items()},
                **{f"fresh_test_mc_{key}": value for key, value in metric(mc).items()},
            }
            test_rows.append(row)
            print(json.dumps({"event": "mc_test", **row}), flush=True)

        del y_val, y_test
        torch.cuda.empty_cache()

    write_csv(output / "mc_candidates.csv", candidate_rows)
    write_csv(output / "fresh_test_results.csv", test_rows)
    completion = {
        "complete": True,
        "seed": seed,
        "source": str(source),
        "training_performed": False,
        "validation_selection_only": True,
        "fresh_test_observation_base": TEST_OBSERVATION_BASE,
        "fresh_test_recorruption_base": TEST_RECORRUPTION_BASE,
        "risk_estimand": "finite-K MC reconstruction risk conditional on each frozen trained model",
        "validation_mc_samples": K_VAL,
        "test_mc_samples": MC_J,
    }
    (output / "complete.json").write_text(json.dumps(completion, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("mc_centered"))
    parser.add_argument("--device", default="cuda:1")
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
