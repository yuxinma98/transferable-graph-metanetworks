"""
Generate ScaleGMN-compatible train/val/test split JSON files for our INR datasets.

Our directory structure:
    data/{mnist,fmnist}_inrs/w{W}/{prefix}_png_train/{digit}/*.pth
    data/{mnist,fmnist}_inrs/w{W}/{prefix}_png_test/{digit}/*.pth

The label is extracted from the digit subdirectory name (p.parent.name),
not from the split directory name (which is how the upstream ScaleGMN
generate_data_splits.py handles the Navon et al. dataset format).

Produces ScaleGMN-compatible JSON format:
    {
        "train": {"path": [...abs paths...], "label": [...str digits...]},
        "val":   {"path": [...], "label": [...]},
        "test":  {"path": [...], "label": [...]}
    }

Usage
-----
# For a single width:
python scripts/generate_splits.py --widths 32

# For Fashion-MNIST:
python scripts/generate_splits.py --dataset fmnist --widths 16 24 32 48

# For multiple widths:
python scripts/generate_splits.py --widths 32 64 128
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent


def generate_split_for_width(
    width: int,
    data_root: Path,
    val_size: int = 5000,
    train_size: int | None = None,
    test_size: int | None = None,
    name: str = "mnist_splits.json",
    seed: int = 42,
    suffix: str = "",
) -> None:
    inr_path = data_root / f"w{width}{suffix}"
    if not inr_path.exists():
        print(f"[WARN] {inr_path} does not exist, skipping.")
        return

    data_split: dict = defaultdict(lambda: defaultdict(list))

    for p in sorted(inr_path.glob("*_png_*/**/*.pth")):
        # Determine train/test split from directory name
        if "train" in p.parts[-3]:  # e.g., mnist_png_train
            s = "train"
        else:
            s = "test"
        # Label is the digit subdirectory name
        label = p.parent.name  # e.g., "3"
        data_split[s]["path"].append(str(p.resolve()))
        data_split[s]["label"].append(label)

    if not data_split["train"]["path"]:
        print(f"[WARN] No training files found under {inr_path}, skipping.")
        return

    # Carve out validation set from training data
    train_indices, val_indices = train_test_split(
        range(len(data_split["train"]["path"])), test_size=val_size, random_state=seed
    )

    # Optionally subsample train
    if train_size is not None and train_size < len(train_indices):
        rng = __import__("random").Random(seed)
        train_indices = rng.sample(list(train_indices), train_size)

    # Optionally subsample test
    test_paths  = data_split["test"]["path"]
    test_labels = data_split["test"]["label"]
    if test_size is not None and test_size < len(test_paths):
        rng = __import__("random").Random(seed)
        test_idx = rng.sample(range(len(test_paths)), test_size)
        test_paths  = [test_paths[i]  for i in test_idx]
        test_labels = [test_labels[i] for i in test_idx]

    result = {
        "train": {
            "path": [data_split["train"]["path"][i] for i in train_indices],
            "label": [data_split["train"]["label"][i] for i in train_indices],
        },
        "val": {
            "path": [data_split["train"]["path"][i] for i in val_indices],
            "label": [data_split["train"]["label"][i] for i in val_indices],
        },
        "test": {
            "path": test_paths,
            "label": test_labels,
        },
    }

    save_path = inr_path / name
    with open(save_path, "w") as f:
        json.dump(result, f)

    n_train = len(result["train"]["path"])
    n_val = len(result["val"]["path"])
    n_test = len(result["test"]["path"])
    print(f"  w={width}: {n_train} train / {n_val} val / {n_test} test -> {save_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ScaleGMN split JSONs for INR datasets")
    p.add_argument("--dataset", choices=["mnist", "fmnist"], default="mnist",
                   help="Which dataset's INRs to process")
    p.add_argument("--widths", type=int, nargs="+", default=[32], help="Hidden widths to process")
    _data_root = os.environ.get("ANYDIM_DATA_ROOT", str(PROJECT_ROOT / "data"))
    p.add_argument("--data-root", type=Path, default=None, help="Data root (auto from dataset name)")
    p.add_argument("--val-size",   type=int, default=5000)
    p.add_argument("--train-size", type=int, default=None, help="Cap training samples per width (default: all)")
    p.add_argument("--test-size",  type=int, default=None, help="Cap test samples per width (default: all)")
    p.add_argument("--name", type=str, default=None, help="Split filename (default: {dataset}_splits.json)")
    p.add_argument("--suffix", type=str, default="", help="Directory suffix after w{width}, e.g. '_canon'")
    args = p.parse_args()
    if args.data_root is None:
        subdir = "fmnist_inrs" if args.dataset == "fmnist" else "mnist_inrs"
        args.data_root = Path(_data_root) / subdir
    if args.name is None:
        prefix = "fmnist" if args.dataset == "fmnist" else "mnist"
        args.name = f"{prefix}_splits.json"
    return args


def main() -> None:
    args = parse_args()
    print(f"Generating splits under {args.data_root}")
    for width in args.widths:
        generate_split_for_width(
            width, args.data_root,
            val_size=args.val_size,
            train_size=args.train_size,
            test_size=args.test_size,
            name=args.name,
            suffix=args.suffix,
        )
    print("Done.")


if __name__ == "__main__":
    main()
