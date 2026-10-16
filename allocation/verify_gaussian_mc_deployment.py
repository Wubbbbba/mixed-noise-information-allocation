"""Verify the finite-K deployment risk for the Gaussian OLS model.

Training uses n independent pairs
    Z = X + N + sqrt(t) M,
    T = X + N - M/sqrt(t),
and the no-intercept OLS coefficient.  At deployment the same trained
coefficient is applied to K recorruptions of one observation Y=X+N and the
outputs are averaged.  For a linear estimator this equals
    b_hat * (Y + sqrt(t) * mean_j M_j).

The script evaluates the exact risk, finds every positive stationary point
from the derivative cubic, checks the K=1 identity, and performs an
independent Monte Carlo check over training repetitions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def components(t: float | np.ndarray, k: float, q: float):
    a = k + q + q * t
    b = k / a
    r = k * q * (1.0 + t) / a
    v = q * (1.0 + 1.0 / t)
    h = r + v
    return a, b, r, v, h


def deployment_risk(t: float | np.ndarray, n: int, k: float, q: float, mc_count: float):
    if n <= 2:
        return np.full_like(np.asarray(t, dtype=float), np.inf)
    a, b, _, _, h = components(t, k, q)
    a_mc = k + q + (0.0 if math.isinf(mc_count) else q * t / mc_count)
    return k - 2.0 * k * b + a_mc * (b * b + h / (a * (n - 2)))


def single_input_risk(t: float | np.ndarray, n: int, k: float, q: float):
    _, _, r, v, _ = components(t, k, q)
    return r + (r + v) / (n - 2)


def derivative_cubic(n: int, k: float, q: float, mc_count: float) -> np.ndarray:
    """Coefficients of a positive multiple of d R_{n,K}/dt.

    The polynomial is written in the scale-free SNR s=k/q and m=n-2.
    Its positive real roots contain every finite stationary point.
    """
    s = k / q
    m = n - 2
    if math.isinf(mc_count):
        return np.array(
            [
                2 * m * s**2 - s**2 - 2 * s - 1,
                s**3 - s**2 - 5 * s - 3,
                -3 * (s + 1) ** 2,
                -(s + 1) ** 3,
            ],
            dtype=float,
        )
    return np.array(
        [
            2 * mc_count * m * s**2
            - mc_count * s**2
            - 2 * mc_count * s
            - mc_count
            - m * s**2
            + 2 * s**2
            + 2 * s,
            mc_count * s**3
            - mc_count * s**2
            - 5 * mc_count * s
            - 3 * mc_count
            + m * s**3
            + m * s**2
            + 2 * s**2
            + 2 * s,
            -3 * mc_count * (s + 1) ** 2,
            -mc_count * (s + 1) ** 3,
        ],
        dtype=float,
    )


def global_optimum(n: int, k: float, q: float, mc_count: float):
    roots = np.roots(derivative_cubic(n, k, q, mc_count))
    candidates = [
        float(z.real)
        for z in roots
        if abs(z.imag) <= 1e-9 * max(1.0, abs(z.real)) and z.real > 0
    ]
    finite = [(t, float(deployment_risk(t, n, k, q, mc_count))) for t in candidates]
    boundary = k if math.isinf(mc_count) else k + (k + q) / (mc_count * (n - 2))
    if not finite or boundary <= min(r for _, r in finite):
        return math.inf, boundary, roots
    return min(finite, key=lambda z: z[1]) + (roots,)


def simulate_training_risk(
    t: float,
    n: int,
    k: float,
    q: float,
    mc_count: float,
    repeats: int,
    seed: int,
    batch: int = 50_000,
):
    rng = np.random.default_rng(seed)
    values = []
    a_mc = k + q + (0.0 if math.isinf(mc_count) else q * t / mc_count)
    remaining = repeats
    while remaining:
        size = min(batch, remaining)
        x = rng.normal(0.0, math.sqrt(k), size=(size, n))
        noise = rng.normal(0.0, math.sqrt(q), size=(size, n))
        aux = rng.normal(0.0, math.sqrt(q), size=(size, n))
        z = x + noise + math.sqrt(t) * aux
        target = x + noise - aux / math.sqrt(t)
        b_hat = np.sum(z * target, axis=1) / np.sum(z * z, axis=1)
        # Exact conditional risk on an independent deployment observation.
        values.append(k - 2.0 * k * b_hat + a_mc * b_hat * b_hat)
        remaining -= size
    values = np.concatenate(values)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(repeats))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("output/experiments/gaussian_mc_deployment_20261010"))
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    if args.n <= 2 or args.k <= 0 or args.q <= 0:
        raise ValueError("require n>2 and k,q>0")
    args.out.mkdir(parents=True, exist_ok=True)

    mc_counts = [1.0, 5.0, 10.0, math.inf]
    t_single, _, _ = global_optimum(args.n, args.k, args.q, 1.0)
    rows = []
    for index, mc_count in enumerate(mc_counts):
        t_opt, risk_opt, roots = global_optimum(args.n, args.k, args.q, mc_count)
        if not math.isfinite(t_opt):
            raise RuntimeError("chosen audit setting unexpectedly has only a boundary infimum")
        sim_mean, sim_se = simulate_training_risk(
            t_opt, args.n, args.k, args.q, mc_count, args.repeats, args.seed + index
        )
        rows.append(
            {
                "K": "infinity" if math.isinf(mc_count) else int(mc_count),
                "optimal_t": t_opt,
                "minimum_exact_risk": risk_opt,
                "risk_at_K1_optimum": float(deployment_risk(t_single, args.n, args.k, args.q, mc_count)),
                "mc_mean_at_optimum": sim_mean,
                "mc_standard_error": sim_se,
                "mc_z_score": (sim_mean - risk_opt) / sim_se,
                "positive_stationary_roots": [
                    float(z.real)
                    for z in roots
                    if abs(z.imag) <= 1e-9 * max(1.0, abs(z.real)) and z.real > 0
                ],
            }
        )
    pd.DataFrame([{k: v for k, v in row.items() if k != "positive_stationary_roots"} for row in rows]).to_csv(
        args.out / "gaussian_mc_optima.csv", index=False
    )

    grid = np.geomspace(0.03, 20.0, 600)
    curve_rows = []
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for mc_count in mc_counts:
        risk = deployment_risk(grid, args.n, args.k, args.q, mc_count)
        label = r"$K=\infty$" if math.isinf(mc_count) else rf"$K={int(mc_count)}$"
        ax.plot(grid, risk, label=label)
        curve_rows.extend(
            {"t": float(t), "K": "infinity" if math.isinf(mc_count) else int(mc_count), "exact_risk": float(r)}
            for t, r in zip(grid, risk)
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"allocation coordinate $t$")
    ax.set_ylabel(r"deployment risk $R_{n,K}(t)$")
    ax.set_title(rf"Gaussian OLS deployment risk ($n={args.n}, k=q=1$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(args.out / "gaussian_mc_risk_curves.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(curve_rows).to_csv(args.out / "gaussian_mc_risk_curves.csv", index=False)

    check_grid = np.geomspace(0.02, 50.0, 1000)
    identity_errors = []
    for mc_count in mc_counts:
        lhs = deployment_risk(check_grid, args.n, args.k, args.q, mc_count)
        rhs = single_input_risk(check_grid, args.n, args.k, args.q)
        a, b, _, _, h = components(check_grid, args.k, args.q)
        eb2 = b * b + h / (a * (args.n - 2))
        reduction = args.q * check_grid * (1.0 - (0.0 if math.isinf(mc_count) else 1.0 / mc_count)) * eb2
        identity_errors.append(float(np.max(np.abs(lhs - (rhs - reduction)))))

    protocol = {
        "model": "Gaussian scalar no-intercept OLS with K-recorruption deployment averaging",
        "n": args.n,
        "k": args.k,
        "q": args.q,
        "training_repetitions": args.repeats,
        "seed": args.seed,
        "simulation": "independent training pairs; exact conditional risk on an independent deployment observation",
        "maximum_identity_absolute_error": max(identity_errors),
        "rows": rows,
    }
    (args.out / "protocol_and_results.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
