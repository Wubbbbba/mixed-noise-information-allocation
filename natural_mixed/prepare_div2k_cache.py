"""Build the DIV2K cache used by the reported neural-network experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


PREPROCESS_SEED = 20261009
TRAIN_IMAGES = 650
VALIDATION_IMAGES = 75
TEST_IMAGES = 75
TRAIN_PATCHES_PER_IMAGE = 4
TRAIN_CROP = 64
EVALUATION_CROP = 128


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def train_patches(
    files: list[Path], rng: np.random.Generator
) -> tuple[torch.Tensor, list[dict[str, int | str]]]:
    patches: list[torch.Tensor] = []
    coordinates: list[dict[str, int | str]] = []
    for path in files:
        array = load_rgb(path)
        height, width, _ = array.shape
        if height < TRAIN_CROP or width < TRAIN_CROP:
            raise ValueError(f"Image is smaller than {TRAIN_CROP}x{TRAIN_CROP}: {path}")
        for patch_index in range(TRAIN_PATCHES_PER_IMAGE):
            top = int(rng.integers(0, height - TRAIN_CROP + 1))
            left = int(rng.integers(0, width - TRAIN_CROP + 1))
            patch = array[top : top + TRAIN_CROP, left : left + TRAIN_CROP]
            patches.append(torch.from_numpy(patch.copy()).permute(2, 0, 1))
            coordinates.append(
                {
                    "file": path.name,
                    "patch": patch_index,
                    "top": top,
                    "left": left,
                }
            )
    return torch.stack(patches), coordinates


def center_crops(files: list[Path]) -> torch.Tensor:
    crops: list[torch.Tensor] = []
    for path in files:
        array = load_rgb(path)
        height, width, _ = array.shape
        if height < EVALUATION_CROP or width < EVALUATION_CROP:
            raise ValueError(
                f"Image is smaller than {EVALUATION_CROP}x{EVALUATION_CROP}: {path}"
            )
        top = (height - EVALUATION_CROP) // 2
        left = (width - EVALUATION_CROP) // 2
        crop = array[top : top + EVALUATION_CROP, left : left + EVALUATION_CROP]
        crops.append(torch.from_numpy(crop.copy()).permute(2, 0, 1))
    return torch.stack(crops)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--div2k-root",
        type=Path,
        required=True,
        help="directory containing DIV2K training images 0001.png through 0800.png",
    )
    parser.add_argument("--output", type=Path, default=Path("div2k_cache.pt"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("div2k_cache_manifest.json")
    )
    args = parser.parse_args()

    root = args.div2k_root.resolve()
    files = [root / f"{index:04d}.png" for index in range(1, 801)]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing DIV2K files: {', '.join(missing[:10])}")

    train_files = files[:TRAIN_IMAGES]
    validation_files = files[TRAIN_IMAGES : TRAIN_IMAGES + VALIDATION_IMAGES]
    test_files = files[TRAIN_IMAGES + VALIDATION_IMAGES :]
    rng = np.random.default_rng(PREPROCESS_SEED)
    train, coordinates = train_patches(train_files, rng)
    cache = {
        "train": train,
        "val": center_crops(validation_files),
        "test": center_crops(test_files),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output)
    manifest = {
        "source_directory": str(root),
        "source_files": {
            "first": files[0].name,
            "last": files[-1].name,
            "count": len(files),
        },
        "splits": {
            "train": [train_files[0].name, train_files[-1].name],
            "validation": [validation_files[0].name, validation_files[-1].name],
            "test": [test_files[0].name, test_files[-1].name],
        },
        "train_patches_per_image": TRAIN_PATCHES_PER_IMAGE,
        "train_crop": TRAIN_CROP,
        "evaluation_center_crop": EVALUATION_CROP,
        "preprocess_seed": PREPROCESS_SEED,
        "train_patch_coordinates": coordinates,
        "tensor_shapes": {key: list(value.shape) for key, value in cache.items()},
        "tensor_dtypes": {key: str(value.dtype) for key, value in cache.items()},
        "tensor_sha256": {key: tensor_sha256(value) for key, value in cache.items()},
        "cache_file_sha256": file_sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: list(value.shape) for key, value in cache.items()}))


if __name__ == "__main__":
    main()
