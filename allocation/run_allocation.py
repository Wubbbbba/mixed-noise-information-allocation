"""Select auxiliary scale using raw mean-plus-SD, then test independently.

Run from any directory. No clean labels enter training or the raw-J selection.
LEGACY NOTATION: locked artifacts use ``J``, ``val_J``, and ``raw_J`` for the
mean-plus-SD stability objective denoted by J_stab in the manuscript. These
serialized field names remain unchanged so existing results stay readable.

The explicit clean-validation oracle is only a diagnostic comparator.
All reported sampling units are whole independent images, never pixels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import t as student_t
import scipy
import torch
from torch import nn

from noise_models import make_pair, sample_observation, second_moment_score

LOW, HIGH = 0.25, 2.0
LOSS_BOUND = (HIGH - LOW) ** 2
SPLITS = {"train": 11, "val": 22, "test": 33}
MODELS = {"pg": 1, "gg": 2, "gp": 3}


def seed_for(base, rep, stream, model=0):
    return int(base + 100000 * rep + 1000 * model + stream)


def sample_images(n, size, rng):
    u = rng.normal(size=(n, 1, size, size))
    u = gaussian_filter(u, sigma=(0, 0, 1.6, 1.6), mode="reflect")
    lo, hi = u.min(axis=(2, 3), keepdims=True), u.max(axis=(2, 3), keepdims=True)
    return LOW + (HIGH - LOW) * (u - lo) / (hi - lo)


class BoundedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1), nn.ReLU(),
            nn.Conv2d(12, 12, 3, padding=1), nn.ReLU(),
            nn.Conv2d(12, 1, 3, padding=1),
        )

    def forward(self, z):
        return LOW + (HIGH - LOW) * torch.sigmoid(self.net((z - 1.125) / 0.875))


def predict(model, z, batch_size=128):
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            model(torch.from_numpy(np.ascontiguousarray(z[i:i+batch_size], dtype=np.float32))).numpy()
            for i in range(0, len(z), batch_size)
        ]).astype(np.float64)


def train(z, target, args, init_seed):
    torch.manual_seed(init_seed)
    model = BoundedCNN()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    zt = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))
    tt = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))
    generator = torch.Generator().manual_seed(init_seed + 1)
    model.train()
    history = []
    for _ in range(args.epochs):
        order = torch.randperm(len(zt), generator=generator)
        total = 0.0
        for ix in order.split(args.batch_size):
            optim.zero_grad(set_to_none=True)
            pred = model(zt[ix])
            loss = (pred - tt[ix]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite raw training loss; labels are not clipped.")
            loss.backward()
            optim.step()
            total += loss.item() * len(ix)
        history.append(total / len(zt))
    return model, history


def score_arrays(pred, x, y, target, noise, params):
    axes = tuple(range(1, pred.ndim))
    raw = np.mean((pred - target) ** 2, axis=axes)
    centered = np.mean(pred**2 - 2 * pred * target, axis=axes)
    clean = np.mean((pred - x) ** 2, axis=axes)
    moment = second_moment_score(y, noise, **params).mean(axis=axes)
    result = {"raw": raw, "centered": centered, "clean": clean, "unbiased": centered + moment}
    if not np.all(np.isfinite(pred)) or not all(np.all(np.isfinite(v)) for v in result.values()):
        raise FloatingPointError("Non-finite predictions or evaluation scores")
    return result


def summary(a):
    mean, sd = float(np.mean(a)), float(np.std(a, ddof=1))
    return {"mean": mean, "sd": sd, "J": mean + sd}


def intervals(a, delta=0.05, bounded=False):
    n = len(a)
    mean = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n))
    radius = float(student_t.ppf(1 - delta / 2, n - 1) * se)
    result = {"mean": mean, "se": se, "t_low": mean - radius, "t_high": mean + radius}
    if bounded:
        if np.any(a < -1e-12) or np.any(a > LOSS_BOUND + 1e-10):
            raise AssertionError("Bounded loss assumption failed")
        hr = LOSS_BOUND * np.sqrt(np.log(2 / delta) / (2 * n))
        result.update(hoeffding_low=max(0.0, mean-hr), hoeffding_high=min(LOSS_BOUND, mean+hr))
        # Maurer & Pontil (2009), Theorem 4, applied to Q/B and 1-Q/B,
        # failure probability delta/2 for each side. s^2 uses ddof=1.
        log_term = np.log(4 / delta)
        er = np.sqrt(2 * np.var(a, ddof=1) * log_term / n) + 7 * LOSS_BOUND * log_term / (3 * (n-1))
        result.update(bernstein_low=max(0.0, mean-er), bernstein_high=min(LOSS_BOUND, mean+er))
    return result


def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def quadrature_order(args, tau):
    # Smaller scales have narrower features in the OU integral.
    return max(args.quad_order, 48) if tau <= 0.25 else args.quad_order


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for key, value in list(row.items()):
            if key in {"model", "method"}:
                continue
            if value in {"True", "False"}:
                row[key] = value == "True"
            else:
                row[key] = float(value)
                if key in {"rep", "test_n", "quad_order"}:
                    row[key] = int(row[key])
    return rows


def run(args):
    out = Path(args.out).resolve()
    if args.resume:
        if not (out / "protocol.json").exists():
            raise ValueError("--resume requires an existing protocol.json")
        previous_protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
        for key, value in vars(args).items():
            if key not in {"resume", "out"} and previous_protocol.get(key) != value:
                raise ValueError(f"Resume changes frozen protocol field {key}")
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / "scores").mkdir()
        (out / "checkpoints").mkdir()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    params = {"gamma": args.gamma, "sigma": args.sigma, "ell": args.ell}
    provenance = vars(args).copy()
    provenance.update(
        torch=str(torch.__version__), numpy=np.__version__, scipy=scipy.__version__,
        python=platform.python_version(), platform=platform.platform(), device="cpu",
        signal_range=[LOW, HIGH], loss_unit="mean squared error per image",
        selection="raw mean plus sample SD on independent validation images",
        test_risk="single auxiliary-corrupted input Z_tau; not direct Y deployment",
        gg_label="deterministic quadrature approximation; no exact-unbiasedness claim for finite quadrature",
        gg_quadrature_orders={str(tau): quadrature_order(args, tau) for tau in args.taus},
        ci_estimand="risk of each fixed selected model, conditional on train/validation",
        ci_confidence="95% pointwise; t intervals asymptotic; bounded Hoeffding and empirical Bernstein finite-sample",
    )
    provenance["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (Path(__file__), Path(__file__).with_name("noise_models.py"))}
    if args.resume:
        event = {"utc": datetime.now(timezone.utc).isoformat(),
                 "source_sha256": provenance["source_sha256"],
                 "note": "Resume missing candidates/groups after interrupted execution; frozen protocol and existing results preserved."}
        with (out / "resume_history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
        candidates = read_csv_rows(out / "candidates.csv")
        selections = read_csv_rows(out / "selections.csv")
        tests = read_csv_rows(out / "test_results.csv")
    else:
        (out / "protocol.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        candidates, selections, tests = [], [], []
    started = time.perf_counter()
    for rep in range(args.reps):
        for noise in args.models:
            if sum(r["rep"] == rep and r["model"] == noise for r in tests) == 5:
                print(json.dumps({"event": "resume_skip_completed", "model": noise, "rep": rep}), flush=True)
                continue
            # Rebuild a partially written test group as a whole, deterministically.
            selections = [r for r in selections if not (r["rep"] == rep and r["model"] == noise)]
            tests = [r for r in tests if not (r["rep"] == rep and r["model"] == noise)]
            model_id = MODELS[noise]
            data = {}
            for split, n in (("train", args.train_n), ("val", args.val_n)):
                x = sample_images(n, args.size, np.random.default_rng(seed_for(args.seed, rep, SPLITS[split])))
                yrng = np.random.default_rng(seed_for(args.seed, rep, SPLITS[split] + 100, model_id))
                y = sample_observation(x, noise, yrng, **params)
                data[split] = (x, y)
            local, checkpoints = [], {}
            for tau in args.taus:
                existing = [r for r in candidates if r["rep"] == rep and r["model"] == noise and r["tau"] == tau]
                if existing:
                    if len(existing) != 1:
                        raise ValueError("Duplicate candidate rows")
                    cp = out / "checkpoints" / f"{noise}_r{rep}_tau{tau:g}.pt"
                    score_path = out / "scores" / f"{noise}_r{rep}_tau{tau:g}_val.npz"
                    if not cp.exists() or not score_path.exists():
                        raise ValueError("Existing candidate is missing a checkpoint or score file")
                    checkpoints[tau] = cp
                    local.append(existing[0])
                    continue
                start = time.perf_counter()
                pairs = {}
                for split, (x, y) in data.items():
                    rng = np.random.default_rng(seed_for(args.seed, rep, SPLITS[split]+200, model_id))
                    pairs[split] = make_pair(y, noise, tau, rng, quad_order=quadrature_order(args, tau), **params)
                pair_seconds = time.perf_counter() - start
                start = time.perf_counter()
                net, history = train(*pairs["train"], args, seed_for(args.seed, rep, 777))
                train_seconds = time.perf_counter() - start
                xv, yv = data["val"]
                zv, tv = pairs["val"]
                pred = predict(net, zv)
                scores = score_arrays(pred, xv, yv, tv, noise, params)
                raw = summary(scores["raw"])
                row = {"model": noise, "rep": rep, "tau": tau,
                       "val_raw_mean": raw["mean"], "val_raw_sd": raw["sd"], "val_J": raw["J"],
                       "val_centered_mean": float(scores["centered"].mean()),
                       "val_clean_mse_diagnostic": float(scores["clean"].mean()),
                       "train_raw_loss_final": history[-1],
                       "max_abs_train_label": float(np.max(np.abs(pairs["train"][1]))),
                       "max_abs_val_label": float(np.max(np.abs(tv))),
                       "pair_seconds": pair_seconds, "train_seconds": train_seconds,
                       "quad_order": quadrature_order(args, tau) if noise == "gg" else 0}
                candidates.append(row)
                local.append(row)
                tag = f"{noise}_r{rep}_tau{tau:g}"
                cp = out / "checkpoints" / (tag + ".pt")
                torch.save(net.state_dict(), cp)
                checkpoints[tau] = cp
                np.savez_compressed(out / "scores" / (tag + "_val.npz"), **scores,
                                    history=np.array(history))
                save_csv(out / "candidates.csv", candidates)
                print(json.dumps({"event": "candidate", **row}, ensure_ascii=False), flush=True)
            # Lock choices before drawing any test observations or labels.
            choices = {
                "raw_J": min(local, key=lambda r: r["val_J"])["tau"],
                "raw_mean": min(local, key=lambda r: r["val_raw_mean"])["tau"],
                "centered_mean": min(local, key=lambda r: r["val_centered_mean"])["tau"],
                "fixed_tau_1": min(args.taus, key=lambda a: abs(a - 1.0)),
                "clean_val_oracle": min(local, key=lambda r: r["val_clean_mse_diagnostic"])["tau"],
            }
            for method, tau in choices.items():
                selections.append({"model": noise, "rep": rep, "method": method, "tau": tau,
                                   "boundary": tau in (min(args.taus), max(args.taus))})
            save_csv(out / "selections.csv", selections)
            xt = sample_images(args.test_n, args.size, np.random.default_rng(seed_for(args.seed, rep, SPLITS["test"])))
            yt = sample_observation(xt, noise,
                    np.random.default_rng(seed_for(args.seed, rep, SPLITS["test"] + 100, model_id)), **params)
            cache = {}
            for tau in sorted(set(choices.values())):
                rng = np.random.default_rng(seed_for(args.seed, rep, SPLITS["test"] + 200, model_id))
                zt, tt = make_pair(yt, noise, tau, rng, quad_order=quadrature_order(args, tau), **params)
                net = BoundedCNN()
                net.load_state_dict(torch.load(checkpoints[tau], map_location="cpu", weights_only=True))
                sc = score_arrays(predict(net, zt), xt, yt, tt, noise, params)
                cache[tau] = sc
                tag = f"{noise}_r{rep}_tau{tau:g}_test"
                np.savez_compressed(out / "scores" / (tag + ".npz"), **sc)
            for method, tau in choices.items():
                sc = cache[tau]
                row = {"model": noise, "rep": rep, "method": method, "tau": tau, "test_n": args.test_n}
                for key, value in intervals(sc["clean"], bounded=True).items():
                    row["clean_" + key] = value
                for key, value in intervals(sc["unbiased"]).items():
                    row["unbiased_" + key] = value
                rs = summary(sc["raw"])
                row.update(test_raw_mean=rs["mean"], test_raw_sd=rs["sd"], test_J=rs["J"])
                tests.append(row)
                print(json.dumps({"event": "test", **row}, ensure_ascii=False), flush=True)
            save_csv(out / "test_results.csv", tests)
    elapsed = time.perf_counter() - started
    (out / "complete.json").write_text(json.dumps({"elapsed_seconds_this_invocation": elapsed,
             "resumed": args.resume,
             "candidate_fits": len(candidates), "selected_results": len(tests)}, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "out": str(out), "elapsed_seconds": elapsed}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--models", nargs="+", default=["pg", "gg", "gp"], choices=["pg", "gg", "gp"])
    p.add_argument("--taus", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--train-n", type=int, default=384)
    p.add_argument("--val-n", type=int, default=512)
    p.add_argument("--test-n", type=int, default=2048)
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--ell", type=float, default=4.0)
    p.add_argument("--quad-order", type=int, default=24)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--resume", action="store_true", help="Continue an interrupted run without changing its protocol")
    run(p.parse_args())
