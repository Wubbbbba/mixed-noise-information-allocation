"""Reproducible numerical checks of allocation-study pair constructions.

This is a construction check, not a training experiment.  Monte Carlo
standard errors are diagnostics and are not used as pass/fail assertions.
GG quadrature is checked on ordinary simulated data and a separately
labelled, non-random stress grid.  Run from the project root, for example:

    python experiments/allocation/verify_constructions.py --n 30000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.integrate import quad

if __package__:
    from .noise_models import (
        label_from_yz, make_pair, sample_observation, second_moment_score,
    )
else:
    from noise_models import (
        label_from_yz, make_pair, sample_observation, second_moment_score,
    )


PARAMETERS = {"gamma": 0.25, "sigma": 0.25, "ell": 4.0}
TAUS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
INTENSITIES = (0.25, 1.0, 2.0)


def _mc_summary(values: np.ndarray, target: float = 0.0) -> dict:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    mcse = sd / math.sqrt(values.size)
    return {
        "n": int(values.size), "estimate": mean, "target": target,
        "error": mean - target, "empirical_mcse": mcse,
        "error_over_mcse": (mean - target) / mcse if mcse > 0 else None,
        "nominal_clt_95_interval": [mean - 1.96 * mcse, mean + 1.96 * mcse],
        "max_absolute_sample": float(np.max(np.abs(values))),
    }


def conditional_moment_checks(n: int, seed: int) -> list[dict]:
    results = []
    for model_index, model in enumerate(("pg", "gg", "gp")):
        for intensity_index, x in enumerate(INTENSITIES):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1, model_index, intensity_index]))
            y = sample_observation(np.full(n, x), model, rng, **PARAMETERS)
            z, t = make_pair(y, model, 1.0, rng, quad_order=24, **PARAMETERS)
            probes = {}
            for name, b in (("1", np.ones(n)), ("tanh(Z)", np.tanh(z)), ("sin(Z)", np.sin(z))):
                entry = _mc_summary((t - x) * b)
                entry["mean_T_b"] = float(np.mean(t * b))
                entry["x_mean_b"] = float(x * np.mean(b))
                probes[name] = entry
            results.append({
                "model": model, "x": x, "tau": 1.0,
                "quad_order": 24 if model == "gg" else None,
                "conditional_cross_moments": probes,
                "second_moment_score": _mc_summary(second_moment_score(y, model, **PARAMETERS), x*x),
                "maximum_absolute": {"y": float(np.max(np.abs(y))), "z": float(np.max(np.abs(z))), "t": float(np.max(np.abs(t)))},
            })
        print(f"conditional moments: {model} complete", flush=True)
    return results


def _pg_second_moment(x: float, tau: float) -> float:
    gamma, sigma = PARAMETERS["gamma"], PARAMETERS["sigma"]
    return math.exp(gamma*gamma/(tau*tau)) * (
        (x + sigma*sigma*gamma/(tau*tau))**2 + gamma*x + sigma*sigma + sigma**4/(tau*tau)
    )


def _pg_fourth_moment(x: float, tau: float) -> float:
    gamma, sigma = PARAMETERS["gamma"], PARAMETERS["sigma"]
    # Exponential tilting of the auxiliary standard Normal by -4*gamma/tau.
    mean = x + 3.0*sigma*sigma*gamma/(tau*tau)
    variance = gamma*x + sigma*sigma + sigma**4/(tau*tau)
    third_cumulant, fourth_cumulant = gamma*gamma*x, gamma**3*x
    return math.exp(6.0*gamma*gamma/(tau*tau)) * (
        mean**4 + 6.0*mean*mean*variance + 4.0*mean*third_cumulant
        + 3.0*variance*variance + fourth_cumulant
    )


def pg_second_moment_checks(n: int, seed: int) -> list[dict]:
    results = []
    gamma, sigma = PARAMETERS["gamma"], PARAMETERS["sigma"]
    for intensity_index, x in enumerate(INTENSITIES):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 2, intensity_index]))
        y = sample_observation(np.full(n, x), "pg", rng, **PARAMETERS)
        normal = rng.standard_normal(n)
        for tau in TAUS:
            t = label_from_yz(y, y + tau*normal, "pg", tau, **PARAMETERS)
            exact = _pg_second_moment(x, tau)
            # Independent scalar integral over the original auxiliary Normal:
            # integrate E_Y[T(Y, tau*u)^2] phi(u) du without using the closed form.
            def integrand(u: float) -> float:
                exponent = -0.5*u*u - 2.0*gamma*u/tau - gamma*gamma/(tau*tau)
                residual_mean = x - sigma*sigma*(tau*u + gamma)/(tau*tau)
                return math.exp(exponent)/math.sqrt(2.0*math.pi) * (gamma*x + sigma*sigma + residual_mean*residual_mean)
            integral, integration_error = quad(integrand, -np.inf, np.inf, epsabs=1e-11, epsrel=1e-11)
            result = _mc_summary(t*t, exact)
            fourth = _pg_fourth_moment(x, tau)
            analytic_mcse = math.sqrt(max(0.0, fourth - exact*exact)/n)
            result.update({
                "x": x, "tau": tau, "analytic_second_moment": exact,
                "analytic_fourth_moment": fourth,
                "analytic_mcse_of_second_moment": analytic_mcse,
                "error_over_analytic_mcse": (result["estimate"]-exact)/analytic_mcse,
                "independent_scalar_integral": float(integral),
                "integrator_error_estimate": float(integration_error),
                "integral_relative_error": abs(integral-exact)/exact,
            })
            results.append(result)
    print("PG second moments and scalar integrals complete", flush=True)
    return results


def _loss_summary(z: np.ndarray, t: np.ndarray) -> dict:
    # A fixed, non-trained predictor taking values strictly between .25 and 2.
    prediction = 1.125 + 0.875*np.tanh(z - 1.125)
    loss = (prediction - t)**2
    mean, sd = float(np.mean(loss)), float(np.std(loss, ddof=1))
    return {"mean": mean, "sd": sd, "mean_plus_sd": mean + sd, "max": float(np.max(loss))}


def _comparison(y: np.ndarray, z: np.ndarray, a: np.ndarray, b: np.ndarray, order_a: int, order_b: int) -> dict:
    absolute = np.abs(a-b)
    relative = absolute/np.maximum(np.abs(b), 1e-12)
    scaled = absolute/(1.0 + np.abs(b))
    worst = int(np.argmax(scaled))
    loss_a, loss_b = _loss_summary(z,a), _loss_summary(z,b)
    loss_changes = {}
    for key in ("mean", "sd", "mean_plus_sd"):
        difference = loss_a[key] - loss_b[key]
        loss_changes[key] = {"difference": difference, "absolute_relative_change": abs(difference)/max(abs(loss_b[key]),1e-12)}
    return {
        "orders": [order_a,order_b],
        "label_absolute_error": {"mean": float(np.mean(absolute)), "p95":float(np.quantile(absolute,.95)), "max":float(np.max(absolute))},
        "label_relative_error_denominator_max_abs_reference_1e_minus_12": {"median":float(np.median(relative)),"p95":float(np.quantile(relative,.95)),"max":float(np.max(relative))},
        "label_scaled_error_denominator_1_plus_abs_reference": {"mean":float(np.mean(scaled)),"p95":float(np.quantile(scaled,.95)),"max":float(np.max(scaled))},
        "worst_scaled_error_sample": {"y":float(y[worst]),"z":float(z[worst]),"label_a":float(a[worst]),"label_b":float(b[worst])},
        "raw_squared_loss_change": loss_changes,
    }


def gg_quadrature_checks(random_n: int, seed: int) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([seed,3]))
    clean = rng.uniform(.25,2.,random_n)
    random_y = sample_observation(clean,"gg",rng,**PARAMETERS)
    random_standardized_eta = rng.standard_t(3.,size=random_n)/math.sqrt(3.)
    # Stress points are deliberately outside a typical simulation sample;
    # they are not assigned a population probability or pooled into the stability objective.
    stress_y, stress_eta = np.meshgrid(
        np.array([-1.,0.,.25,1.,2.,4.,8.,16.]),
        np.array([-20.,-8.,-2.,0.,2.,8.,20.]), indexing="ij",
    )
    stress_y, stress_eta = stress_y.reshape(-1),stress_eta.reshape(-1)
    y = np.concatenate([random_y,stress_y])
    eta = np.concatenate([random_standardized_eta,stress_eta])
    if y.size > 2000:
        raise ValueError("The common and stress GG samples together must not exceed 2000.")
    cases = []
    for tau in TAUS:
        z = y+tau*eta
        labels = {order:label_from_yz(y,z,"gg",tau,quad_order=order,**PARAMETERS) for order in (24,48,96)}
        subsets = {}
        for name, selection in (("random",slice(0,random_n)),("stress",slice(random_n,None))):
            yy, zz = y[selection], z[selection]
            subset_labels = {order:values[selection] for order,values in labels.items()}
            comparisons = [_comparison(yy,zz,subset_labels[a],subset_labels[b],a,b) for a,b in ((24,48),(48,96),(24,96))]
            subsets[name] = {
                "n":int(yy.size),
                "max_absolute_y":float(np.max(np.abs(yy))), "max_absolute_z":float(np.max(np.abs(zz))),
                "per_order": {str(order):{"max_absolute_label":float(np.max(np.abs(values))),"raw_squared_loss":_loss_summary(zz,values)} for order,values in subset_labels.items()},
                "comparisons":comparisons,
            }
        common_24_96 = subsets["random"]["comparisons"][2]
        common_48_96 = subsets["random"]["comparisons"][1]
        scaled_error = common_24_96["label_scaled_error_denominator_1_plus_abs_reference"]["max"]
        stability_change = common_24_96["raw_squared_loss_change"]["mean_plus_sd"]["absolute_relative_change"]
        reference_change = common_48_96["label_scaled_error_denominator_1_plus_abs_reference"]["max"]
        cases.append({
            "tau":tau,"subsets":subsets,
            "random_sample_24_order_diagnostic": {
                "max_scaled_label_error_below_1e_minus_3": scaled_error < 1e-3,
                "relative_raw_stability_objective_change_below_1e_minus_3": stability_change < 1e-3,
                "reference_48_96_max_scaled_error_below_1e_minus_4": reference_change < 1e-4,
                "acceptable_on_this_random_sample": scaled_error < 1e-3 and stability_change < 1e-3 and reference_change < 1e-4,
            },
        })
        print(f"GG quadrature tau={tau}: common max scaled 24/96={scaled_error:.4g}, raw-stability-objective relative change={stability_change:.4g}, 48/96={reference_change:.4g}",flush=True)
    accepted = all(case["random_sample_24_order_diagnostic"]["acceptable_on_this_random_sample"] for case in cases)
    return {
        "random_signal_distribution":"Uniform[0.25,2]",
        "random_sample_count":random_n,"stress_sample_count":int(stress_y.size),
        "stress_y_values":[-1.,0.,.25,1.,2.,4.,8.,16.],
        "stress_auxiliary_noise_divided_by_tau":[-20.,-8.,-2.,0.,2.,8.,20.],
        "predictor":"f(z)=1.125+0.875*tanh(z-1.125), bounded in [0.25,2]",
        "loss":"(f(Z)-T)^2; sample SD uses ddof=1",
        "diagnostic_thresholds_are_numerical_tolerances_not_theorems":True,
        "order_24_accepted_on_all_random_samples":accepted,
        "cases":cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",type=int,default=30000)
    parser.add_argument("--quad-samples",type=int,default=1600)
    parser.add_argument("--seed",type=int,default=20261008)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[2]/"output"/"experiments"/"allocation_checks_20261008.json")
    args = parser.parse_args()
    if args.n < 2 or not 2 <= args.quad_samples <= 1944:
        parser.error("n must be >=2 and quad-samples must lie in [2,1944].")
    started = time.perf_counter()
    results = {
        "seed":args.seed,"n_per_conditional_moment_case":args.n,"parameters":PARAMETERS,
        "versions":{"numpy":np.__version__,"scipy":scipy.__version__},
        "notes":[
            "Pseudo-labels and scores are not clipped; PG and GP are direct formulas, GG uses numerical quadrature.",
            "Cross-moment MCSE is computed from the paired residual (T-x)*b(Z), not from two independent estimated means.",
            "MCSE and nominal CLT intervals are diagnostics, not finite-sample guarantees or pass/fail tests; heavy tails can make empirical MCSE underestimate uncertainty.",
            "PG second-moment uncertainty is additionally computed from the analytic fourth moment.",
            "GG 96-order values are a numerical reference, not an exact oracle. Stress-grid loss statistics are descriptive and are not population risk estimates.",
            "These fixed bounded predictor checks assess numerical construction; they do not prove trained-network allocation performance.",
        ],
    }
    results["conditional_moment_checks"] = conditional_moment_checks(args.n,args.seed)
    results["pg_second_moment_checks"] = pg_second_moment_checks(args.n,args.seed)
    results["gg_quadrature_checks"] = gg_quadrature_checks(args.quad_samples,args.seed)
    results["elapsed_seconds"] = time.perf_counter()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(f"saved {args.output} ({results['elapsed_seconds']:.1f}s)",flush=True)


if __name__ == "__main__":
    main()
