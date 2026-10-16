"""Known-noise observations and finite-variance self-supervised pairs.

The formulas follow the mixed-noise constructions in the manuscript.  PG uses
Gaussian auxiliary noise; GG and GP use the polynomial-tail density
``h_tau(v) = 2*tau**3 / (pi*(tau**2 + v**2)**2)``.  All calculations are
float64 and no pseudo-label clipping is performed.

For GG, the exact OU integral is replaced by tensor-product Gauss-Jacobi /
Gauss-Hermite quadrature.  Conditional unbiasedness is a property of the
exact integral: finite quadrature introduces numerical error, which must
be assessed by increasing ``quad_order`` before interpreting experiments.
The functions constructing labels never receive or use the clean signal.
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln, logsumexp, roots_hermitenorm, roots_jacobi

Model = Literal["pg", "gg", "gp"]
FloatArray = NDArray[np.float64]

# Bound each vectorized work array, including at higher quadrature orders.
_MAX_WORKING_ELEMENTS = 262_144


def _validate_parameters(
    model: str, gamma: float, sigma: float, ell: float, tau: float | None = None
) -> None:
    if model not in ("pg", "gg", "gp"):
        raise ValueError(f"Unknown noise model {model!r}; use 'pg', 'gg', or 'gp'.")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and strictly positive.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative.")
    if not np.isfinite(ell) or ell <= 0:
        raise ValueError("ell must be finite and strictly positive.")
    if tau is not None and (not np.isfinite(tau) or tau <= 0):
        raise ValueError("tau must be finite and strictly positive.")


def _as_finite_array(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _check_result(result: ArrayLike, context: str) -> FloatArray:
    array = np.asarray(result, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(
            f"Nonfinite {context}; inspect the noise parameters and input range. "
            "Pseudo-labels are not clipped."
        )
    return array


def sample_observation(
    x: ArrayLike,
    model: Model,
    rng: np.random.Generator,
    gamma: float = 0.25,
    sigma: float = 0.25,
    ell: float = 4.0,
) -> FloatArray:
    """Draw one observation per clean value, preserving the input shape.

    PG: ``Y = gamma*Pois(X/gamma) + N(0, sigma**2)``.
    GG: ``Y = S + N(0, sigma**2)``, with ``S ~ Gamma(ell, scale=X/ell)``.
    GP: ``Y = gamma*Pois(S/gamma)`` with the same Gamma latent variable.
    The Gamma latent variable is defined to be zero at ``X=0``.
    """
    _validate_parameters(model, gamma, sigma, ell)
    clean = _as_finite_array(x, "x")
    if np.any(clean < 0):
        raise ValueError("x must be nonnegative for all three noise models.")

    if model == "pg":
        observation = gamma * rng.poisson(clean / gamma)
        observation = observation + rng.normal(0.0, sigma, size=clean.shape)
    else:
        latent = rng.gamma(shape=ell, scale=clean / ell)
        if model == "gg":
            observation = latent + rng.normal(0.0, sigma, size=clean.shape)
        else:
            observation = gamma * rng.poisson(latent / gamma)
    return _check_result(observation, "observation")


@lru_cache(maxsize=32)
def _gg_quadrature(ell: float, order: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Flatten nodes/weights for ell*r**(ell-1) dr times standard Normal dU."""
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
        raise ValueError("quad_order must be a positive integer.")
    # Jacobi weight (1-s)**0 * (1+s)**(ell-1), r=(s+1)/2.
    # Normalizing the positive weights is equivalent to multiplying by
    # ell/2**ell, and avoids an unnecessary exponential in ell.
    nodes_r, weights_r = roots_jacobi(int(order), 0.0, ell - 1.0)
    nodes_r = (nodes_r + 1.0) / 2.0
    weights_r = weights_r / np.sum(weights_r)
    # Probabilists' Hermite rule integrates exp(-U**2/2).
    nodes_u, weights_u = roots_hermitenorm(int(order))
    weights_u = weights_u / np.sum(weights_u)

    r = np.repeat(nodes_r, order)
    gaussian_shift = (
        np.sqrt(1.0 - nodes_r * nodes_r)[:, None] * nodes_u[None, :]
    ).reshape(-1)
    weights = (weights_r[:, None] * weights_u[None, :]).reshape(-1)
    for array in (r, gaussian_shift, weights):
        array.setflags(write=False)
    return r, gaussian_shift, weights


def _gg_label(
    y: FloatArray, z: FloatArray, tau: float, sigma: float, ell: float, order: int
) -> FloatArray:
    r, gaussian_shift, weights = _gg_quadrature(float(ell), order)
    y_flat, z_flat = y.reshape(-1), z.reshape(-1)
    result = np.empty(y_flat.size, dtype=np.float64)
    block_size = max(1, _MAX_WORKING_ELEMENTS // r.size)

    for start in range(0, y_flat.size, block_size):
        stop = min(start + block_size, y_flat.size)
        yy, zz = y_flat[start:stop, None], z_flat[start:stop, None]
        v = zz - yy * r[None, :] - sigma * gaussian_shift[None, :]
        hyp = np.hypot(tau, v)
        # Normalizing constants cancel: h(v)/h(z-y) = [hypot(tau,z-y)/hyp]^4.
        ratio = np.exp(4.0 * (np.log(np.hypot(tau, zz - yy)) - np.log(hyp)))
        # d_y h(v) = -r*h'(v) = 4*r*v*h(v)/(tau**2+v**2).
        # Thus (y*u - sigma**2*d_y u)/h(z-y) has the MINUS sign below.
        score = yy - (4.0 * sigma * sigma) * r[None, :] * (v / hyp) / hyp
        result[start:stop] = np.sum(ratio * score * weights[None, :], axis=1)
    return result.reshape(y.shape)


def _gp_counts(y: FloatArray, gamma: float) -> NDArray[np.int64]:
    scaled = y / gamma
    rounded = np.rint(scaled)
    if np.any(y < 0) or not np.allclose(scaled, rounded, rtol=1e-10, atol=1e-7):
        raise ValueError("GP observations must be nonnegative integer multiples of gamma.")
    if np.any(rounded >= np.iinfo(np.int64).max):
        raise ValueError("GP count exceeds the supported int64 range.")
    return rounded.astype(np.int64)


def _gp_label(y: FloatArray, z: FloatArray, tau: float, gamma: float, ell: float) -> FloatArray:
    counts = _gp_counts(y, gamma).reshape(-1)
    z_flat, y_flat = z.reshape(-1), y.reshape(-1)
    result = np.zeros(counts.size, dtype=np.float64)

    for n_value in np.unique(counts):
        n = int(n_value)
        if n == 0:
            continue
        indices = np.flatnonzero(counts == n)
        # Handle the finite k-sum in chunks too, so very large counts do not
        # force an unbounded work matrix.  All summands are nonnegative.
        k_block_size = min(n, _MAX_WORKING_ELEMENTS)
        row_block_size = max(1, _MAX_WORKING_ELEMENTS // k_block_size)
        log_prefactor = (
            np.log(ell) + np.log(gamma) + gammaln(n + 1.0) - gammaln(n + ell)
        )
        for row_start in range(0, indices.size, row_block_size):
            current = indices[row_start : row_start + row_block_size]
            zz = z_flat[current, None]
            log_den_hyp = np.log(np.hypot(tau, zz - y_flat[current, None]))
            accumulated = np.full(current.size, -np.inf, dtype=np.float64)
            for k_start in range(0, n, k_block_size):
                k = np.arange(k_start, min(k_start + k_block_size, n), dtype=np.float64)
                log_weights = log_prefactor + gammaln(k + ell) - gammaln(k + 1.0)
                log_ratio = 4.0 * (
                    log_den_hyp - np.log(np.hypot(tau, zz - gamma * k[None, :]))
                )
                subtotal = logsumexp(log_weights[None, :] + log_ratio, axis=1)
                accumulated = np.logaddexp(accumulated, subtotal)
            result[current] = np.exp(accumulated)
    return result.reshape(y.shape)


def label_from_yz(
    y: ArrayLike,
    z: ArrayLike,
    model: Model,
    tau: float,
    gamma: float = 0.25,
    sigma: float = 0.25,
    ell: float = 4.0,
    quad_order: int = 24,
) -> FloatArray:
    """Evaluate a pseudo-label from observed Y and auxiliary input Z only.

    ``y`` and ``z`` may be broadcast-compatible arrays.  PG and GP formulas
    are evaluated directly; only GG depends on ``quad_order``.  Re-evaluate
    the *same* ``y,z`` at larger orders to measure GG quadrature convergence.
    PG assumes Gaussian auxiliary noise with standard deviation ``tau``;
    GG/GP assume ``tau/sqrt(3) * StudentT(df=3)`` auxiliary noise.
    """
    _validate_parameters(model, gamma, sigma, ell, tau)
    observation, auxiliary = np.broadcast_arrays(
        _as_finite_array(y, "y"), _as_finite_array(z, "z")
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if model == "pg":
            eta = auxiliary - observation
            tau_squared = tau * tau
            label = np.exp(-gamma * eta / tau_squared - gamma * gamma / (2.0 * tau_squared))
            label = label * (observation - (sigma * sigma / tau_squared) * (eta + gamma))
        elif model == "gg":
            label = _gg_label(observation, auxiliary, tau, sigma, ell, quad_order)
        else:
            label = _gp_label(observation, auxiliary, tau, gamma, ell)
    return _check_result(label, f"{model.upper()} pseudo-label")


def make_pair(
    y: ArrayLike,
    model: Model,
    tau: float,
    rng: np.random.Generator,
    gamma: float = 0.25,
    sigma: float = 0.25,
    ell: float = 4.0,
    quad_order: int = 24,
) -> tuple[FloatArray, FloatArray]:
    """Generate one (Z,T) pair from each observation, without clean values.

    In all three models the auxiliary noise has variance ``tau**2``.
    At fixed positive parameters, the exact labels have finite second
    moments for bounded clean signals; GG numerical integration should be
    convergence-checked separately.  Negative PG/GG labels are retained.
    """
    _validate_parameters(model, gamma, sigma, ell, tau)
    observation = _as_finite_array(y, "y")
    if model == "pg":
        eta = rng.normal(0.0, tau, size=observation.shape)
    else:
        eta = (tau / np.sqrt(3.0)) * rng.standard_t(df=3.0, size=observation.shape)
    z = _check_result(observation + eta, "auxiliary input")
    t = label_from_yz(observation, z, model, tau, gamma, sigma, ell, quad_order)
    return z, t


def second_moment_score(
    y: ArrayLike,
    model: Model,
    gamma: float = 0.25,
    sigma: float = 0.25,
    ell: float = 4.0,
) -> FloatArray:
    """Return U(Y) with E[U(Y)|X]=X**2 under the stated observation model.

    Together with an independent evaluation pair this gives the corrected
    score ``f(Z)**2 - 2*f(Z)*T + U(Y)``, whose expectation is clean MSE.
    U(Y) can be negative.  It is neither a squared residual nor a quantity
    to clip.  Unbiasedness for a trained predictor requires evaluation
    observations independent of its training data.
    """
    _validate_parameters(model, gamma, sigma, ell)
    observation = _as_finite_array(y, "y")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if model == "pg":
            score = observation * observation - gamma * observation - sigma * sigma
        elif model == "gg":
            score = (ell / (ell + 1.0)) * (observation * observation - sigma * sigma)
        else:
            _gp_counts(observation, gamma)
            score = (ell / (ell + 1.0)) * (observation * observation - gamma * observation)
    return _check_result(score, "second-moment score")


__all__ = ["sample_observation", "make_pair", "label_from_yz", "second_moment_score"]
