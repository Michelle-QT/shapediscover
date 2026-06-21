"""One-off downloader for the standard image benchmarks (needs network).

Pulls MNIST / Fashion-MNIST / CIFAR-10 via torchvision and saves the FULL
``<name>_X.npy`` / ``<name>_y.npy`` into ``examples/datasets/`` (no subsampling;
the loaders subsample on demand via an ``n`` kwarg, so the full data is available
when wanted). Run once (with network), from the repo root::

    python -m benchmarks.download_image_datasets

Idempotent: skips a dataset whose .npy files already exist. The .npy files are
large (MNIST/Fashion ~180 MB, CIFAR-10 ~600 MB) and gitignored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "examples" / "datasets"


def _save(name: str, X: np.ndarray, y: np.ndarray) -> None:
    np.save(OUT / f"{name}_X.npy", X.astype(np.float32))
    np.save(OUT / f"{name}_y.npy", y.astype(np.int64))
    print(f"  saved {name}: X {X.shape}, y {y.shape} (full)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import torchvision

    cache = OUT / "_torchvision_cache"
    specs = [
        ("mnist", torchvision.datasets.MNIST),
        ("fashion_mnist", torchvision.datasets.FashionMNIST),
        ("cifar10", torchvision.datasets.CIFAR10),
    ]
    for name, cls in specs:
        if (OUT / f"{name}_X.npy").exists():
            print(f"  {name}: already present, skipping")
            continue
        print(f"  downloading {name} ...")
        ds = cls(root=str(cache), train=True, download=True)
        data = ds.data
        X = np.asarray(data).reshape(len(data), -1).astype(np.float32) / 255.0
        y = np.asarray(ds.targets)
        _save(name, X, y)
    print("done.")


if __name__ == "__main__":
    main()
