"""Monte Carlo audit of the finite-sample Gaussian no-intercept OLS result.

The theoretical risk is evaluated exactly, while the reported simulation uses
independent training pairs and independent test samples.  No CNN, pseudo-label
quadrature, or clean-label selection is involved here.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def risk_theory(n: int, k: float, q: float, t: float) -> float:
    if n <= 2:
        return float("inf")
    a = k + q + q * t
    r = k * q * (1.0 + t) / a
    v = q * (1.0 + 1.0 / t)
    return r + (r + v) / (n - 2)


def t_star(n: int, k: float, q: float) -> float | None:
    denominator = np.sqrt(n - 1.0) * k - q
    return (k + q) / denominator if denominator > 0.0 else None


def simulate_one(n: int, k: float, q: float, t: float, repetitions: int,
                 test_per_fit: int, rng: np.random.Generator) -> tuple[float, float]:
    # The two training coordinates are generated independently for every fit.
    x = np.sqrt(k) * rng.normal(size=(repetitions, n))
    noise = np.sqrt(q) * rng.normal(size=(repetitions, n))
    aux = np.sqrt(q * t) * rng.normal(size=(repetitions, n))
    z = x + noise + aux
    # ``aux`` is generated as sqrt(q*t) * N(0,1) = sqrt(t) M.
    # The construction is T_t = X + N - M/sqrt(t), hence the
    # corresponding term is aux / t (not aux / sqrt(t)).
    target = x + noise - aux / t
    bhat = np.sum(z * target, axis=1) / np.sum(z * z, axis=1)

    xt = np.sqrt(k) * rng.normal(size=(repetitions, test_per_fit))
    nt = np.sqrt(q) * rng.normal(size=(repetitions, test_per_fit))
    mt = np.sqrt(q) * rng.normal(size=(repetitions, test_per_fit))
    zt = xt + nt + np.sqrt(t) * mt
    losses = (bhat[:, None] * zt - xt) ** 2
    per_fit = losses.mean(axis=1)
    return float(per_fit.mean()), float(per_fit.std(ddof=1) / np.sqrt(repetitions))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    q = 1.0
    # Internal, critical, and boundary cases at the same sample size.
    cases = [
        {"case": "internal_n8", "n": 8, "k": 1.0, "q": q},
        {"case": "critical_n8", "n": 8, "k": 1.0 / np.sqrt(7.0), "q": q},
        {"case": "boundary_n8", "n": 8, "k": 0.35, "q": q},
        {"case": "internal_n16", "n": 16, "k": 1.0, "q": q},
        {"case": "internal_n32", "n": 32, "k": 1.0, "q": q},
        {"case": "internal_n64", "n": 64, "k": 1.0, "q": q},
    ]
    # The fixed log-grid is only a visualization/check; the exact stationary
    # point is also simulated separately below.
    grid = np.geomspace(0.0625, 32.0, 33)
    rows, summaries = [], []
    for case_id, case in enumerate(cases):
        n, k = int(case["n"]), float(case["k"])
        exact = t_star(n, k, q)
        values = sorted(set(float(t) for t in grid.tolist() + ([exact] if exact else [])))
        rng = np.random.default_rng(args.seed + 1000 * case_id)
        for t in values:
            mc, se = simulate_one(n, k, q, t, args.repetitions, args.test_per_fit, rng)
            rows.append({"case": case["case"], "n": n, "k": k, "q": q, "t": t,
                         "tau": np.sqrt(q * t), "theory_risk": risk_theory(n, k, q, t),
                         "mc_risk": mc, "mc_se": se,
                         "relative_error": (mc - risk_theory(n, k, q, t)) / risk_theory(n, k, q, t)})
        if exact is None:
            eval_t = float(values[-1])
            kind = "boundary_infimum"
            infimum = ((n - 1) * k + q) / (n - 2)
        else:
            eval_t = exact
            kind = "finite_interior"
            infimum = None
        theory_at_eval = risk_theory(n, k, q, eval_t)
        mc, se = simulate_one(n, k, q, eval_t, args.repetitions, args.test_per_fit, rng)
        summaries.append({"case": case["case"], "n": n, "k": k, "q": q,
                          "condition_value": np.sqrt(n - 1.0) * k - q,
                          "t_star": exact, "tau_star": np.sqrt(q * exact) if exact else None,
                          "classification": kind, "theory_at_evaluation": theory_at_eval,
                          "predicted_infimum": infimum,
                          "evaluation_t": eval_t, "mc_risk_at_evaluation": mc,
                          "mc_se_at_evaluation": se,
                          "mc_minus_theory_at_evaluation": mc - theory_at_eval})

    asymptotic = []
    for n in [8, 16, 32, 64, 128, 256, 512, 1024]:
        ts = t_star(n, 1.0, 1.0)
        tau = np.sqrt(ts) if ts else None
        asymptotic.append({"n": n, "t_star": ts, "tau_star": tau,
                           "n_quarter_tau": tau * n ** 0.25,
                           "asymptotic_constant": np.sqrt(2.0)})

    write_csv(root / "gaussian_finite_sample_curves.csv", rows)
    write_csv(root / "gaussian_finite_sample_summary.csv", summaries)
    write_csv(root / "gaussian_tau_asymptotic.csv", asymptotic)
    metadata = {"seed": args.seed, "repetitions": args.repetitions,
                "test_per_fit": args.test_per_fit, "cases": cases,
                "n_le_2": "E[1/chi2_n]=infinity, hence expected OLS risk is infinite for every finite t",
                "tau_asymptotic": "sqrt(q*(k+q)/k)*n^(-1/4) for fixed k,q>0"}
    (root / "gaussian_finite_sample_protocol.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, case in zip(axes.flat, cases):
        rr = [r for r in rows if r["case"] == case["case"]]
        ax.plot([r["t"] for r in rr], [r["theory_risk"] for r in rr], "k-", label="theory")
        ax.errorbar([r["t"] for r in rr], [r["mc_risk"] for r in rr],
                    yerr=[1.96 * r["mc_se"] for r in rr], fmt="o", ms=3,
                    alpha=.7, label="MC 95% SE")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(case["case"]); ax.set_xlabel("t"); ax.set_ylabel("risk")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.savefig(root / "gaussian_finite_sample_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"curves": len(rows), "summaries": len(summaries),
                      "out": str(root)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/experiments/gaussian_finite_sample_20261009")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--repetitions", type=int, default=20000)
    parser.add_argument("--test-per-fit", type=int, default=16)
    main(parser.parse_args())
