"""Monte Carlo check of the Gamma--Poisson negative-binomial thinning formulas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_gp_thinning_comparison import thinning_pair


def run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    rows = []
    for pi in args.pi_values:
        latent = rng.gamma(shape=args.ell, scale=args.x / args.ell, size=args.samples)
        counts = rng.poisson(latent / args.gamma)
        y = args.gamma * counts
        z, target = thinning_pair(y, pi, args.gamma, args.ell, rng)
        theoretical_variance_z = (args.gamma * args.x + args.x**2 / args.ell) / pi
        theoretical_variance_target = (args.gamma * args.x + args.x**2 / args.ell) / (1.0 - pi)
        rows.append({
            "pi": pi,
            "mean_z": float(z.mean()),
            "mean_target": float(target.mean()),
            "theoretical_mean": args.x,
            "variance_z": float(z.var(ddof=1)),
            "theoretical_variance_z": theoretical_variance_z,
            "variance_target": float(target.var(ddof=1)),
            "theoretical_variance_target": theoretical_variance_target,
            "conditional_covariance_z_target_mc": float(np.cov(z, target, ddof=1)[0, 1]),
            "mean_z_standard_errors": float((z.mean() - args.x) / np.sqrt(theoretical_variance_z / args.samples)),
            "mean_target_standard_errors": float((target.mean() - args.x) / np.sqrt(theoretical_variance_target / args.samples)),
        })
    payload = {
        "seed": args.seed,
        "samples_per_pi": args.samples,
        "x": args.x,
        "gamma": args.gamma,
        "ell": args.ell,
        "rows": rows,
        "note": "Covariance is conditional on the fixed x simulation and should be zero; numerical agreement checks but does not replace the beta-binomial proof.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/experiments/gp_thinning_comparison_20261010/moment_check.json")
    parser.add_argument("--seed", type=int, default=20261010405)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--x", type=float, default=1.25)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--ell", type=float, default=4.0)
    parser.add_argument(
        "--pi-values", "--r-values", dest="pi_values", nargs="+", type=float,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="GP thinning proportion pi; --r-values is retained as a legacy alias.",
    )
    run(parser.parse_args())
