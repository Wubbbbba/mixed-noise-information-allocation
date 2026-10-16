"""Natural-image mixed-noise experiment reported in the paper.

The script trains a standard residual DnCNN on DIV2K patches and evaluates
validation-selected allocation rules on external natural-image benchmarks.
Test images are not loaded until all validation selections for a
noise/model/seed group are saved.

PG, GG and GP use the paper's original auxiliary-noise/kernel-ratio pairs.
GG uses finite Gauss-Jacobi/Gauss-Hermite quadrature and therefore remains a
numerical approximation.  Gamma--Poisson thinning is evaluated separately and
is not substituted for the main three-noise protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path

# Use the pure-Python protobuf backend to avoid binary-version conflicts in
# environments where PyTorch imports ONNX while constructing optimizers.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
from PIL import Image
import scipy
from scipy.special import roots_hermitenorm, roots_jacobi
import torch
from torch import nn
import torch.nn.functional as F


LOW = 0.25
HIGH = 2.0
DYNAMIC_RANGE = HIGH - LOW
TAUS = [0.25, 0.50, 1.0, 2.0, 4.0, 8.0]
DATASETS = ["B100", "Set14", "Urban100"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(x) for x in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


class DnCNN(nn.Module):
    def __init__(self, channels: int = 3, depth: int = 17, width: int = 64):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(inplace=True)
        ]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(width, channels, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the paper's physical intensity scale [0.25, 2], while feeding
        # the conventional normalized representation to DnCNN.
        normalized = (x - LOW) / DYNAMIC_RANGE
        return LOW + DYNAMIC_RANGE * (normalized - self.body(normalized))


BASE_NOISE_PARAMETERS = {"gamma": 0.250, "sigma": 0.250, "ell": 4.0}


def observe(x: torch.Tensor, kind: str, params: dict[str, float], seed: int) -> torch.Tensor:
    seed_all(seed)
    gamma, sigma, ell = params["gamma"], params["sigma"], params["ell"]
    if kind == "pg":
        return gamma * torch.poisson(x.clamp_min(0) / gamma) + sigma * torch.randn_like(x)
    latent = torch._standard_gamma(torch.full_like(x, ell)) * x.clamp_min(0) / ell
    if kind == "gg":
        return latent + sigma * torch.randn_like(x)
    return gamma * torch.poisson(latent.clamp_min(0) / gamma)


def gg_nodes(ell: float, order: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    r, wr = roots_jacobi(order, 0.0, ell - 1.0)
    r = (r + 1.0) / 2.0
    wr = wr / wr.sum()
    u, wu = roots_hermitenorm(order)
    wu = wu / wu.sum()
    rr = np.repeat(r, order)
    shift = (np.sqrt(1.0 - r * r)[:, None] * u[None, :]).reshape(-1)
    weight = (wr[:, None] * wu[None, :]).reshape(-1)
    return tuple(torch.as_tensor(v, dtype=torch.float32, device=device) for v in (rr, shift, weight))


def student_t3_like(y: torch.Tensor) -> torch.Tensor:
    """Draw an exact Student-t(3) tensor without the slow Distribution API."""
    numerator = torch.randn_like(y)
    chi_square = torch.zeros_like(y)
    for _ in range(3):
        chi_square.add_(torch.randn_like(y).square())
    return numerator / torch.sqrt(chi_square / 3.0)


def gg_label(
    y: torch.Tensor,
    z: torch.Tensor,
    tau: float,
    sigma: float,
    ell: float,
    order: int,
    block: int = 32768,
) -> torch.Tensor:
    r, shift, weight = gg_nodes(ell, order, y.device)
    yf, zf = y.flatten(), z.flatten()
    out = torch.empty_like(yf)
    tau2 = tau * tau
    for lo in range(0, yf.numel(), block):
        hi = min(yf.numel(), lo + block)
        yy, zz = yf[lo:hi, None], zf[lo:hi, None]
        v = zz - yy * r[None, :] - sigma * shift[None, :]
        den = tau2 + v.square()
        ratio = ((tau2 + (zz - yy).square()) / den).square()
        score = yy - 4.0 * sigma * sigma * r[None, :] * v / den
        out[lo:hi] = (ratio * score * weight[None, :]).sum(1)
    return out.reshape_as(y)


def gp_label(
    y: torch.Tensor,
    z: torch.Tensor,
    tau: float,
    gamma: float,
    ell: float,
    max_work: int = 2_000_000,
) -> torch.Tensor:
    """Evaluate the original finite-sum Gamma--Poisson kernel-ratio label."""
    counts = torch.round(y / gamma).to(torch.int64).flatten()
    if torch.any(counts < 0) or not torch.allclose(
        y, counts.reshape_as(y).to(y.dtype) * gamma, rtol=1e-5, atol=2e-5
    ):
        raise ValueError("Gamma--Poisson observations must be nonnegative count multiples")
    yf, zf = y.flatten().double(), z.flatten().double()
    result = torch.zeros_like(yf)
    for n_tensor in torch.unique(counts):
        n = int(n_tensor)
        if n == 0:
            continue
        indices = torch.nonzero(counts == n_tensor, as_tuple=False).flatten()
        k_block = min(n, max_work)
        row_block = max(1, max_work // k_block)
        log_prefactor = (
            math.log(ell)
            + math.log(gamma)
            + math.lgamma(n + 1.0)
            - math.lgamma(n + ell)
        )
        for row_start in range(0, indices.numel(), row_block):
            current = indices[row_start : row_start + row_block]
            zz = zf[current, None]
            log_den = torch.log(torch.hypot(torch.full_like(zz, tau), zz - yf[current, None]))
            accumulated = torch.full((current.numel(),), -torch.inf, dtype=torch.float64, device=y.device)
            for k_start in range(0, n, k_block):
                k = torch.arange(k_start, min(k_start + k_block, n), dtype=torch.float64, device=y.device)
                log_weights = (
                    log_prefactor
                    + torch.lgamma(k + ell)
                    - torch.lgamma(k + 1.0)
                )
                log_ratio = 4.0 * (
                    log_den
                    - torch.log(torch.hypot(torch.full_like(zz, tau), zz - gamma * k[None, :]))
                )
                subtotal = torch.logsumexp(log_weights[None, :] + log_ratio, dim=1)
                accumulated = torch.logaddexp(accumulated, subtotal)
            result[current] = torch.exp(accumulated)
    result = result.reshape_as(y).float()
    if not torch.isfinite(result).all():
        raise FloatingPointError("non-finite Gamma--Poisson pseudo-label")
    return result


def make_pair(
    y: torch.Tensor,
    kind: str,
    allocation: float,
    params: dict[str, float],
    seed: int,
    quad_order: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed_all(seed)
    gamma, sigma, ell = params["gamma"], params["sigma"], params["ell"]
    if kind == "pg":
        tau = allocation
        eta = tau * torch.randn_like(y)
        z = y + eta
        t = torch.exp(-gamma * eta / tau**2 - gamma**2 / (2 * tau**2))
        t = t * (y - (sigma**2 / tau**2) * (eta + gamma))
        return z, t
    if kind == "gg":
        tau = allocation
        eta = (tau / math.sqrt(3.0)) * student_t3_like(y)
        z = y + eta
        return z, gg_label(y, z, tau, sigma, ell, quad_order)
    tau = allocation
    eta = (tau / math.sqrt(3.0)) * student_t3_like(y)
    z = y + eta
    return z, gp_label(y, z, tau, gamma, ell)


def allocations(kind: str, override: list[float] | None) -> list[float]:
    if override:
        return override
    return TAUS


def train_model(
    z: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    init_seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[DnCNN, list[float]]:
    seed_all(init_seed)
    model = DnCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator(device=device).manual_seed(init_seed + 1)
    steps = math.ceil(len(z) / batch_size) * epochs
    tail: list[float] = []
    model.train()
    for step in range(steps):
        ix = torch.randint(0, len(z), (batch_size,), device=device, generator=generator)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(z[ix])
            loss = F.mse_loss(pred, target[ix])
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        if step >= steps - min(200, steps):
            tail.append(float(loss.detach()))
    return model, tail


def validation_scores(
    model: DnCNN, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor, batch: int = 8
) -> dict[str, float]:
    model.eval()
    raw, q, clean = [], [], []
    with torch.no_grad():
        for lo in range(0, len(x), batch):
            xx, zz, tt = x[lo:lo+batch], z[lo:lo+batch], t[lo:lo+batch]
            with torch.autocast("cuda", dtype=torch.float16):
                f = model(zz)
            f = f.float().clamp(LOW, HIGH)
            dims = (1, 2, 3)
            raw.append(((f - tt).square()).mean(dims).cpu())
            q.append((f.square() - 2 * f * tt).mean(dims).cpu())
            clean.append(((f - xx).square()).mean(dims).cpu())
    ra, qa, ca = (torch.cat(v).double().numpy() for v in (raw, q, clean))
    return {
        "val_raw_mean": float(ra.mean()),
        "val_raw_sd": float(ra.std(ddof=1)),
        "val_raw_J": float(ra.mean() + ra.std(ddof=1)),
        "val_q_mean": float(qa.mean()),
        "val_q_sd": float(qa.std(ddof=1)),
        "val_clean_mse": float(ca.mean()),
        "val_pooled_psnr": float(10 * np.log10(DYNAMIC_RANGE**2 / ca.mean())),
    }


def load_benchmark(root: Path, name: str) -> list[tuple[str, torch.Tensor]]:
    paths = sorted((root / name / "x2").glob("*.png"))
    if not paths:
        raise FileNotFoundError(root / name / "x2")
    out = []
    for path in paths:
        with Image.open(path) as image:
            a = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            a = LOW + DYNAMIC_RANGE * a
        out.append((path.name, torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)))
    return out


def test_input(
    y: torch.Tensor,
    kind: str,
    allocation: float,
    params: dict[str, float],
    seed: int,
) -> torch.Tensor:
    seed_all(seed)
    if kind == "pg":
        return y + allocation * torch.randn_like(y)
    if kind == "gg":
        return y + allocation * student_t3_like(y) / math.sqrt(3.0)
    return y + allocation * student_t3_like(y) / math.sqrt(3.0)


def metric_row(
    errors: list[float],
    ssims: list[float],
    forward_ms: list[float],
    pixel_sse: float,
    pixel_n: int,
) -> dict[str, float | int | str]:
    a = np.asarray(errors, dtype=np.float64)
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    radius = 1.96 * sd / math.sqrt(len(a))
    pixel_mse = pixel_sse / pixel_n
    return {
        "image_mean_mse": mean,
        "image_mean_mse_ci_low": max(0.0, mean - radius),
        "image_mean_mse_ci_high": mean + radius,
        "equal_image_pooled_psnr": float(10 * np.log10(DYNAMIC_RANGE**2 / mean)),
        "pixel_pooled_mse": float(pixel_mse),
        "pixel_pooled_psnr": float(10 * np.log10(DYNAMIC_RANGE**2 / pixel_mse)),
        "mean_image_psnr": float(np.mean(10 * np.log10(DYNAMIC_RANGE**2 / np.maximum(a, 1e-12)))),
        "mean_image_ssim": float(np.mean(ssims)),
        "mean_forward_ms": float(np.mean(forward_ms[1:] if len(forward_ms) > 1 else forward_ms)),
        "mse_interval_scope": "normal interval across independent test images for one fixed trained model",
        "n_images": len(a),
    }


def image_ssim(x: torch.Tensor, estimate: torch.Tensor) -> float:
    """Standard 11x11 Gaussian-window SSIM, averaged over RGB and pixels."""
    coordinates = torch.arange(11, device=x.device, dtype=torch.float32) - 5.0
    kernel_1d = torch.exp(-0.5 * (coordinates / 1.5).square())
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :])[None, None]
    kernel = kernel.expand(x.shape[1], 1, 11, 11)

    def smooth(value: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(value, (5, 5, 5, 5), mode="reflect"), kernel, groups=value.shape[1])

    x = x.float()
    estimate = estimate.float()
    mu_x, mu_e = smooth(x), smooth(estimate)
    sigma_x = (smooth(x.square()) - mu_x.square()).clamp_min(0)
    sigma_e = (smooth(estimate.square()) - mu_e.square()).clamp_min(0)
    sigma_xe = smooth(x * estimate) - mu_x * mu_e
    c1 = (0.01 * DYNAMIC_RANGE) ** 2
    c2 = (0.03 * DYNAMIC_RANGE) ** 2
    score = ((2 * mu_x * mu_e + c1) * (2 * sigma_xe + c2)) / (
        (mu_x.square() + mu_e.square() + c1) * (sigma_x + sigma_e + c2)
    )
    return float(score.mean())


def evaluate_external(
    model: DnCNN,
    dataset: list[tuple[str, torch.Tensor]],
    kind: str,
    allocation: float,
    params: dict[str, float],
    seed: int,
    device: torch.device,
    save_path: Path,
) -> dict[str, float | int | str]:
    model.eval()
    errors, ssims, forward_ms, names = [], [], [], []
    pixel_sse, pixel_n = 0.0, 0
    with torch.no_grad():
        for index, (name, xcpu) in enumerate(dataset):
            x = xcpu.to(device)
            y = observe(x, kind, params, stable_seed(seed, "test_y", kind, index))
            z = test_input(y, kind, allocation, params, stable_seed(seed, "test_z", kind, allocation, index))
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                f = model(z)
            torch.cuda.synchronize(device)
            forward_ms.append(1000.0 * (time.perf_counter() - start))
            f = f.float().clamp(LOW, HIGH)
            diff = (f - x).double()
            mse = float(diff.square().mean())
            errors.append(mse)
            ssims.append(image_ssim(x, f))
            names.append(name)
            pixel_sse += float(diff.square().sum())
            pixel_n += diff.numel()
    np.savez_compressed(
        save_path,
        names=np.asarray(names),
        mse=np.asarray(errors),
        ssim=np.asarray(ssims),
        forward_ms=np.asarray(forward_ms),
    )
    result = metric_row(errors, ssims, forward_ms, pixel_sse, pixel_n)
    result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--models", nargs="+", default=["pg", "gg", "gp"])
    parser.add_argument("--modes", nargs="+", default=["selfsup"])
    parser.add_argument("--allocations", nargs="+", type=float)
    parser.add_argument("--train-count", type=int, default=650)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--quad-train", type=int, default=8)
    parser.add_argument("--quad-val", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "per_image").mkdir(exist_ok=True)
    params = dict(BASE_NOISE_PARAMETERS)

    cache = torch.load(args.data_cache, map_location="cpu")
    xtrain = cache["train"][::4][:args.train_count].float().div_(255)
    xval = cache["val"].float().div_(255)
    xtrain = (LOW + DYNAMIC_RANGE * xtrain).to(device)
    xval = (LOW + DYNAMIC_RANGE * xval).to(device)
    protocol = {
        **vars(args),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_cache_sha256": file_sha256(Path(args.data_cache)),
        "data_cache_tensor_shapes": {
            key: list(value.shape) for key, value in cache.items()
        },
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "hardware": {"gpu": torch.cuda.get_device_name(device)},
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "architecture": "residual DnCNN depth=17 width=64 with BatchNorm",
        "train_source": "one fixed 64x64 crop from each of the first train_count DIV2K training images",
        "validation": "75 held-out DIV2K 128x128 center crops",
        "external_test": DATASETS,
        "evaluation_input": "f_allocation(Z_allocation), not f(Y)",
        "physical_intensity_range": [LOW, HIGH],
        "psnr_dynamic_range": DYNAMIC_RANGE,
        "selection": "validation only; test datasets loaded after selection JSON is saved",
        "gg_note": "finite quadrature approximation; no exact-unbiasedness claim",
        "gp_note": "original Student-t(3) auxiliary and finite kernel-ratio sum",
        "noise_parameters": params,
    }
    protocol["data_cache"] = str(Path(args.data_cache).resolve())
    protocol["benchmark_root"] = str(Path(args.benchmark_root).resolve())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    candidate_rows: list[dict] = []
    test_rows: list[dict] = []
    for kind_index, kind in enumerate(args.models):
        grid = allocations(kind, args.allocations)
        ytrain = observe(xtrain, kind, params, stable_seed(args.seed, kind, "train_y"))
        yval = observe(xval, kind, params, stable_seed(args.seed, kind, "val_y"))
        rows_by_mode: dict[str, list[dict]] = {mode: [] for mode in args.modes}
        for allocation in grid:
            pair_seed_train = stable_seed(args.seed, kind, allocation, "train_pair")
            pair_seed_val = stable_seed(args.seed, kind, allocation, "val_pair")
            ztrain, ttrain = make_pair(ytrain, kind, allocation, params, pair_seed_train, args.quad_train)
            zval, tval = make_pair(yval, kind, allocation, params, pair_seed_val, args.quad_val)
            for mode in args.modes:
                tag = f"{kind}_{mode}_a{allocation:g}"
                checkpoint = out / "checkpoints" / f"{tag}.pt"
                result_json = out / f"{tag}.json"
                if checkpoint.exists() and result_json.exists():
                    row = json.loads(result_json.read_text(encoding="utf-8"))
                else:
                    target = ttrain if mode == "selfsup" else xtrain
                    start = time.time()
                    model, tail = train_model(
                        ztrain, target, device,
                        stable_seed(args.seed, kind_index, mode, "init"),
                        args.epochs, args.batch_size, args.lr,
                    )
                    scores = validation_scores(model, xval, zval, tval)
                    row = {
                        "noise": kind,
                        "mode": mode,
                        "seed": args.seed,
                        "noise_level": "base",
                        "train_count": args.train_count,
                        "allocation": allocation,
                        "allocation_name": "tau",
                        "train_tail_loss": float(np.mean(tail)),
                        "train_seconds": time.time() - start,
                        "max_abs_target": float(target.abs().max()),
                        **scores,
                    }
                    torch.save(model.state_dict(), checkpoint)
                    result_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
                    del model
                    torch.cuda.empty_cache()
                rows_by_mode[mode].append(row)
                candidate_rows.append(row)
                write_csv(out / "candidates.partial.csv", candidate_rows)
                print(json.dumps({"event": "candidate", **row}), flush=True)
            del ztrain, ttrain, zval, tval
            torch.cuda.empty_cache()

        selections: dict[str, float] = {}
        if "selfsup" in rows_by_mode:
            local = rows_by_mode["selfsup"]
            selections.update(
                q_r=min(local, key=lambda row: row["val_q_mean"])["allocation"],
                raw_J=min(local, key=lambda row: row["val_raw_J"])["allocation"],
                fixed=min(grid, key=lambda value: abs(value - 1.0)),
                clean_val=min(local, key=lambda row: row["val_clean_mse"])["allocation"],
            )
        if "supervised" in rows_by_mode:
            local = rows_by_mode["supervised"]
            selections["supervised_clean"] = min(local, key=lambda row: row["val_clean_mse"])["allocation"]
        selection_path = out / f"{kind}_selection.json"
        selection_path.write_text(json.dumps(selections, indent=2), encoding="utf-8")

        # Test paths are touched only after the validation selections above are durable.
        benchmark_root = Path(args.benchmark_root)
        for dataset_name in DATASETS:
            dataset = load_benchmark(benchmark_root, dataset_name)
            evaluated: dict[tuple[str, float], dict[str, float | int]] = {}
            for method, allocation in selections.items():
                mode = "supervised" if method == "supervised_clean" else "selfsup"
                key = (mode, allocation)
                if key not in evaluated:
                    checkpoint = out / "checkpoints" / f"{kind}_{mode}_a{allocation:g}.pt"
                    model = DnCNN().to(device)
                    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                    evaluated[key] = evaluate_external(
                        model, dataset, kind, allocation, params,
                        stable_seed(args.seed, dataset_name), device,
                        out / "per_image" / f"{kind}_{mode}_a{allocation:g}_{dataset_name}.npz",
                    )
                    del model
                    torch.cuda.empty_cache()
                result = evaluated[key]
                row = {
                    "noise": kind,
                    "method": method,
                    "mode": mode,
                    "seed": args.seed,
                    "noise_level": "base",
                    "train_count": args.train_count,
                    "allocation": allocation,
                    "allocation_name": "tau",
                    "dataset": dataset_name,
                    **result,
                }
                test_rows.append(row)
                write_csv(out / "test_results.partial.csv", test_rows)
                print(json.dumps({"event": "test", **row}), flush=True)
        del ytrain, yval
        torch.cuda.empty_cache()

    write_csv(out / "candidates.csv", candidate_rows)
    write_csv(out / "test_results.csv", test_rows)
    (out / "complete.json").write_text(
        json.dumps({"complete": True, "seed": args.seed, "candidate_rows": len(candidate_rows), "test_rows": len(test_rows)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
