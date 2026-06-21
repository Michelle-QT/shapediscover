"""One-off downloader for the standard image benchmarks (needs network).

Pulls MNIST / Fashion-MNIST / CIFAR-10 via torchvision, subsamples each to a
size our O(n) method handles, and saves ``<name>_X.npy`` / ``<name>_y.npy`` into
``examples/datasets/`` so the benchmark loaders can read them offline afterwards.

Run once (with network), from the repo root::

    python -m benchmarks.download_image_datasets

Idempotent: skips a dataset whose .npy files already exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "examples" / "datasets"
N_SUBSAMPLE = 4000
SEED = 0


def _save(name: str, X: np.ndarray, y: np.ndarray) -> None:
    rng = np.random.default_rng(SEED)
    if len(X) > N_SUBSAMPLE:
        idx = rng.choice(len(X), size=N_SUBSAMPLE, replace=False)
        X, y = X[idx], y[idx]
    np.save(OUT / f"{name}_X.npy", X.astype(np.float32))
    np.save(OUT / f"{name}_y.npy", y.astype(np.int64))
    print(f"  saved {name}: X {X.shape}, y {y.shape}")


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
