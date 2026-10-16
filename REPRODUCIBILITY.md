# Reproducibility guide

This repository separates source code and compact reference tables from large
model artifacts and third-party datasets. The main repository is sufficient to
inspect the method, rebuild the data cache, rerun every reported protocol, and
recompute the manuscript summaries. The optional release archive preserves the
reported natural-image checkpoints and per-image measurements.

## Environment

The GPU experiments were run with Python 3.10.12, an NVIDIA A30 GPU, CUDA
12.5, and the NVIDIA PyTorch 2.4 development build
`2.4.0a0+f70bd71a48.nv24.06`. The recorded numerical environment used NumPy
1.24.4 and SciPy 1.13.1. A portable stable-version environment is provided in
`requirements-reproduction.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-reproduction.txt
```

The original neural-network runs used automatic mixed precision and
`torch.backends.cudnn.benchmark = True`. Consequently, independent GPU reruns
are expected to reproduce the reported selection pattern and aggregate values
up to ordinary floating-point and GPU-kernel variation, rather than byte for
byte. Byte-exact inspection of the reported trained models is provided by the
optional checkpoint archive.

## Data preparation

Obtain the 800 DIV2K training images from the official source and place
`0001.png` through `0800.png` in one directory. Rebuild the exact crop protocol
with:

```bash
python natural_mixed/prepare_div2k_cache.py \
  --div2k-root /path/to/DIV2K_train_HR \
  --output data/div2k_cache.pt \
  --manifest data/div2k_cache_manifest.json
```

The preprocessing seed is `20261009`. Images `0001`--`0650` each contribute
four random 64 by 64 crops. The reported natural-image experiment uses the
first crop from each group. Images `0651`--`0725` provide 128 by 128 center
crops for validation. The remaining 75 center crops are retained for the
controlled GR2R study and are not used as natural-image test data.

Place the HR benchmark images under:

```text
benchmarks/
  B100/x2/*.png
  Set14/x2/*.png
  Urban100/x2/*.png
```

Validate file counts and write a local hash manifest before training:

```bash
python natural_mixed/benchmark_manifest.py \
  --benchmark-root /path/to/benchmarks \
  --output data/benchmark_manifest.json
```

Dataset files are not redistributed because they are maintained by their
original providers.

## Natural-image experiment

Run the three registered training seeds:

```bash
python natural_mixed/run_mixed_dncnn.py --seed 0 --device cuda:0 \
  --out output/natural_mixed/base_seed0 \
  --data-cache data/div2k_cache.pt --benchmark-root /path/to/benchmarks
python natural_mixed/run_mixed_dncnn.py --seed 1 --device cuda:0 \
  --out output/natural_mixed/base_seed1 \
  --data-cache data/div2k_cache.pt --benchmark-root /path/to/benchmarks
python natural_mixed/run_mixed_dncnn.py --seed 2 --device cuda:0 \
  --out output/natural_mixed/base_seed2 \
  --data-cache data/div2k_cache.pt --benchmark-root /path/to/benchmarks
python natural_mixed/summarize_formal.py output/natural_mixed
python natural_mixed/verify_natural_mixed.py --root output/natural_mixed
```

For each seed, the script trains 18 candidates: three mixed-noise models and
six allocation values. Validation selections are written before the external
test images are loaded. The full verification checks the 54 checkpoints, 81
per-image archives, all selection files, and every aggregate table.

## Compact result audit

The public repository retains the manuscript-level tables in
`reference_results`. They can be checked without a GPU or external datasets:

```bash
python -m compileall -q allocation natural_mixed
python allocation/verify_seed_fusion.py \
  --result-dir reference_results/seed_fusion_20261015
python natural_mixed/verify_natural_mixed.py --aggregate-only \
  --root reference_results/natural_mixed_base_20261015/results
```

The controlled mixed-noise, Gaussian finite-sample, finite-deployment,
Gamma--Poisson thinning, supervised-reference, and seed-fusion outputs are
stored in correspondingly named directories under `reference_results`.

## Optional artifact archive

The release file
`natural-image-reproduction-artifacts-v1.zip` contains the 54 reported DnCNN
checkpoints, 81 per-image test archives, run protocols, validation selections,
and aggregate tables. Pass the extracted artifact directory to
`verify_natural_mixed.py` without `--aggregate-only` for an exact audit of the
archived outputs.
