"""Validate the external benchmark layout and record file hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


EXPECTED_COUNTS = {"B100": 100, "Set14": 14, "Urban100": 100}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmark_manifest.json"))
    args = parser.parse_args()

    root = args.benchmark_root.resolve()
    datasets: dict[str, list[dict[str, object]]] = {}
    for name, expected in EXPECTED_COUNTS.items():
        paths = sorted((root / name / "x2").glob("*.png"))
        if len(paths) != expected:
            raise ValueError(f"{name}: expected {expected} PNG files, found {len(paths)}")
        rows: list[dict[str, object]] = []
        for path in paths:
            with Image.open(path) as image:
                width, height = image.size
            rows.append(
                {
                    "file": path.name,
                    "width": width,
                    "height": height,
                    "sha256": sha256(path),
                }
            )
        datasets[name] = rows

    payload = {"benchmark_root": str(root), "datasets": datasets}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({name: len(rows) for name, rows in datasets.items()}))


if __name__ == "__main__":
    main()
