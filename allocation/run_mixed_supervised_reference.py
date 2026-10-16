"""Train input-matched supervised references for the mixed-noise allocation run.

The source run is left untouched.  For every existing (noise, repetition,
tau) candidate this script regenerates the same clean images, observation Y,
and auxiliary input Z_tau, but trains against X.  Selection uses clean
validation MSE only.  No test observations are generated here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
import torch

from noise_models import sample_observation
from run_allocation import HIGH, LOW, MODELS, SPLITS, predict, sample_images, seed_for, train


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def auxiliary_input(y: np.ndarray, noise: str, tau: float, rng: np.random.Generator) -> np.ndarray:
    """Generate exactly the Z_tau marginal used by ``make_pair`` without T."""
    if noise == "pg":
        eta = rng.normal(0.0, tau, size=y.shape)
    elif noise in {"gg", "gp"}:
        eta = (tau / np.sqrt(3.0)) * rng.standard_t(df=3.0, size=y.shape)
    else:
        raise ValueError(noise)
    z = np.asarray(y + eta, dtype=np.float64)
    if not np.all(np.isfinite(z)):
        raise FloatingPointError("nonfinite supervised input")
    return z


def candidate_tag(noise: str, rep: int, tau: float) -> str:
    return f"{noise}_r{rep}_tau{tau:g}"


def main(args: argparse.Namespace) -> None:
    source = Path(args.run).resolve()
    out = Path(args.out).resolve()
    source_protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    expected = int(source_protocol["reps"]) * len(source_protocol["models"]) * len(source_protocol["taus"])
    if len(list((source / "checkpoints").glob("*.pt"))) != expected:
        raise ValueError("source run is incomplete")

    if out.exists() and not args.resume:
        raise FileExistsError(f"output exists; use --resume: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "scores").mkdir(exist_ok=True)
    (out / "candidate_records").mkdir(exist_ok=True)

    protocol_path = out / "protocol.json"
    frozen = {
        "source_run": str(source),
        "source_protocol_sha256": hashlib.sha256((source / "protocol.json").read_bytes()).hexdigest(),
        "purpose": "input-matched clean-label supervised training reference",
        "selection": "minimum clean validation MSE within the same finite tau grid",
        "test_generated_by_this_script": False,
        "estimand": "f_tau(Z_tau)",
        "models": source_protocol["models"],
        "taus": source_protocol["taus"],
        "reps": source_protocol["reps"],
        "seed": source_protocol["seed"],
        "train_n": source_protocol["train_n"],
        "val_n": source_protocol["val_n"],
        "size": source_protocol["size"],
        "epochs": source_protocol["epochs"],
        "batch_size": source_protocol["batch_size"],
        "lr": source_protocol["lr"],
        "gamma": source_protocol["gamma"],
        "sigma": source_protocol["sigma"],
        "ell": source_protocol["ell"],
        "signal_range": [LOW, HIGH],
        "architecture": "same BoundedCNN imported from run_allocation.py",
        "auxiliary": "PG Gaussian sd=tau; GG/GP tau/sqrt(3)*Student-t(df=3)",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)},
        "source_sha256": {
            "this_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "run_allocation.py": hashlib.sha256(Path(__file__).with_name("run_allocation.py").read_bytes()).hexdigest(),
            "noise_models.py": hashlib.sha256(Path(__file__).with_name("noise_models.py").read_bytes()).hexdigest(),
        },
    }
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        for key in ("source_protocol_sha256", "models", "taus", "reps", "seed", "train_n", "val_n", "size", "epochs", "batch_size", "lr", "gamma", "sigma", "ell"):
            if old.get(key) != frozen.get(key):
                raise ValueError(f"resume changes frozen protocol field {key}")
    else:
        protocol_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")

    torch.set_num_threads(int(source_protocol.get("threads", 4)))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    train_args = SimpleNamespace(
        lr=float(source_protocol["lr"]), epochs=int(source_protocol["epochs"]),
        batch_size=int(source_protocol["batch_size"]),
    )
    params = {k: float(source_protocol[k]) for k in ("gamma", "sigma", "ell")}
    rows: list[dict] = []
    started = time.perf_counter()
    for rep in range(int(source_protocol["reps"])):
        for noise in source_protocol["models"]:
            model_id = MODELS[noise]
            data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for split, n in (("train", int(source_protocol["train_n"])), ("val", int(source_protocol["val_n"]))):
                x = sample_images(n, int(source_protocol["size"]), np.random.default_rng(seed_for(int(source_protocol["seed"]), rep, SPLITS[split])))
                y = sample_observation(x, noise, np.random.default_rng(seed_for(int(source_protocol["seed"]), rep, SPLITS[split] + 100, model_id)), **params)
                data[split] = (x, y)
            for tau_value in source_protocol["taus"]:
                tau = float(tau_value)
                tag = candidate_tag(noise, rep, tau)
                record_path = out / "candidate_records" / f"{tag}.json"
                checkpoint = out / "checkpoints" / f"{tag}.pt"
                score_path = out / "scores" / f"{tag}_val.npz"
                if record_path.exists() and checkpoint.exists() and score_path.exists():
                    rows.append(json.loads(record_path.read_text(encoding="utf-8")))
                    print(json.dumps({"event": "resume_skip", "tag": tag}), flush=True)
                    continue
                if record_path.exists() or checkpoint.exists() or score_path.exists():
                    raise ValueError(f"partial candidate state for {tag}")
                z: dict[str, np.ndarray] = {}
                for split, (_x, y) in data.items():
                    z[split] = auxiliary_input(
                        y, noise, tau,
                        np.random.default_rng(seed_for(int(source_protocol["seed"]), rep, SPLITS[split] + 200, model_id)),
                    )
                t0 = time.perf_counter()
                net, history = train(z["train"], data["train"][0], train_args, seed_for(int(source_protocol["seed"]), rep, 777))
                seconds = time.perf_counter() - t0
                prediction = predict(net, z["val"])
                axes = tuple(range(1, prediction.ndim))
                mse = np.mean((prediction - data["val"][0]) ** 2, axis=axes)
                psnr = 10.0 * np.log10((HIGH - LOW) ** 2 / mse)
                record = {
                    "model": noise, "rep": rep, "tau": tau,
                    "val_clean_mse": float(mse.mean()),
                    "val_clean_mse_sd": float(mse.std(ddof=1)),
                    "val_pooled_psnr_db": float(10.0 * np.log10((HIGH - LOW) ** 2 / mse.mean())),
                    "val_mean_per_image_psnr_db": float(psnr.mean()),
                    "train_clean_loss_final": float(history[-1]),
                    "train_seconds": seconds,
                }
                torch.save(net.state_dict(), checkpoint)
                np.savez_compressed(score_path, clean=mse, per_image_psnr_db=psnr)
                record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                rows.append(record)
                save_csv(out / "candidates.csv", sorted(rows, key=lambda r: (r["model"], r["rep"], r["tau"])))
                print(json.dumps({"event": "trained", "tag": tag, "val_mse": record["val_clean_mse"], "seconds": seconds}), flush=True)

    if len(rows) != expected:
        raise AssertionError(f"expected {expected} candidates, found {len(rows)}")
    rows = sorted(rows, key=lambda r: (r["model"], r["rep"], r["tau"]))
    save_csv(out / "candidates.csv", rows)
    selections = []
    for noise in source_protocol["models"]:
        for rep in range(int(source_protocol["reps"])):
            group = [r for r in rows if r["model"] == noise and int(r["rep"]) == rep]
            best = min(group, key=lambda r: (r["val_clean_mse"], r["tau"]))
            selections.append({
                "model": noise, "rep": rep, "method": "supervised_clean_train_val_select",
                "tau": best["tau"], "validation_objective": best["val_clean_mse"],
            })
    save_csv(out / "selections.csv", selections)
    lock = {
        "selection_locked_before_fresh_test": True,
        "selection_file_sha256": hashlib.sha256((out / "selections.csv").read_bytes()).hexdigest(),
        "candidate_count": len(rows), "selection_count": len(selections),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (out / "training_complete.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "out": str(out), **lock}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="output/experiments/allocation_main_20261008")
    parser.add_argument("--out", default="output/experiments/mixed_supervised_reference_20261010")
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
