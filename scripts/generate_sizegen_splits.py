"""
Generate non-overlapping splits for the size-generalization experiment.

Each width gets a disjoint subset of FMNIST images, ensuring no image appears
at multiple widths. This prevents the model from memorizing image identity
across widths.

Layout:
  - w16 (train width): train + val + test
  - w24, w32, w40, w48, w64, w80, w96 (test widths): test only

All widths use different INR fitting seeds (already done during generation —
different random init per width since no seed is set in generate_inrs.py).

Usage:
    python scripts/generate_sizegen_splits.py --dataset fmnist --init-type sp \
        --train-width 16 --test-widths 24 32 40 48 64 80 96 \
        --val-size 1000 --test-size 2000

By default the image identities are read off the already-fitted INRs of the train
width. With --from-dataset they are derived from the source image dataset instead,
which lets the split be written *before* anything is fitted — necessary when only
each width's test subset will ever be fitted (`generate_inrs.py --only-from-split`,
for widths where fitting all 70k images would be wasteful). Combine it with
--create-dirs to lay out the empty width directories at the same time:

    python scripts/generate_sizegen_splits.py --dataset fmnist --init-type sp \
        --from-dataset --create-dirs --train-width 32 \
        --test-widths 48 64 96 128 256 512 1024 --test-size 1000 \
        --split-name fmnist_sizegen_splits_v2.json

Produces one JSON per width in the corresponding w{W}_{suffix}/ directory:
    fmnist_sizegen_splits.json
"""
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from torchvision import datasets

PROJECT_ROOT = Path(__file__).parent.parent


def collect_files(inr_path: Path) -> dict[str, list[Path]]:
    """Collect all .pth files grouped by (split, digit).

    Returns dict keyed by "{split}/{digit}" -> sorted list of paths.
    """
    files = defaultdict(list)
    for p in sorted(inr_path.glob("*_png_*/**/*.pth")):
        split_dir = p.parts[-3]  # e.g., "fmnist_png_train"
        digit = p.parts[-2]      # e.g., "3"
        split = "train" if "train" in split_dir else "test"
        files[f"{split}/{digit}"].append(p)
    return files


def get_image_index(path: Path) -> str:
    """Extract the image index from a .pth filename (e.g., '000042' from '000042.pth')."""
    return path.stem


def dataset_image_ids(dataset: str, image_data_root: Path) -> list[tuple[str, str, str]]:
    """Every image's (orig_split, digit, file stem), read from the source dataset.

    Mirrors `canonical_file_ids()` in generate_inrs.py: that script walks the 60k
    train images then the 10k test images in dataset order and names each file by a
    per-(split, digit) counter, so these triples are exactly the file names any
    width will have — whether or not it has been fitted yet.
    """
    ds_cls = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
    ids: list[tuple[str, str, str]] = []
    counters: dict[tuple[str, int], int] = {}
    for orig_split, is_train in (("train", True), ("test", False)):
        ds = ds_cls(root=image_data_root, train=is_train, download=True)
        for label in ds.targets.tolist():
            key = (orig_split, label)
            idx = counters.get(key, 0)
            counters[key] = idx + 1
            ids.append((orig_split, str(label), f"{idx:06d}"))
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "fmnist"], default="fmnist")
    parser.add_argument("--init-type", type=str, default="sp", help="Suffix for width dirs (e.g. sp, mup, sp_d3, mup_d3)")
    parser.add_argument("--train-width", type=int, default=16)
    parser.add_argument("--test-widths", type=int, nargs="+", default=[24, 32, 40, 48, 64, 80, 96])
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=2000, help="Test size per width (including train width)")
    parser.add_argument("--seed", type=int, default=123, help="Seed for splitting images across widths")
    parser.add_argument("--split-name", type=str, default="fmnist_sizegen_splits.json")
    parser.add_argument(
        "--from-dataset",
        action="store_true",
        help="Derive image identities from the source image dataset instead of from the "
             "train width's fitted INRs, so the split can be written before any fit.",
    )
    parser.add_argument(
        "--create-dirs",
        action="store_true",
        help="Create the empty w{W}_{suffix}/{prefix}_png_{train,test}/{digit}/ tree for widths "
             "that do not exist yet, instead of skipping them.",
    )
    _data_root = os.environ.get("ANYDIM_DATA_ROOT", str(PROJECT_ROOT / "data"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--image-data-root",
        type=Path,
        default=Path(_data_root),
        help="Download root of the source image dataset (used by --from-dataset)",
    )
    args = parser.parse_args()

    suffix = f"_{args.init_type}"
    if args.data_root is None:
        subdir = "fmnist_inrs" if args.dataset == "fmnist" else "mnist_inrs"
        args.data_root = Path(_data_root) / subdir

    all_widths = [args.train_width] + args.test_widths
    n_widths = len(all_widths)

    if args.from_dataset:
        # Identities come from the source dataset, so nothing has to be fitted yet.
        all_images = dataset_image_ids(args.dataset, args.image_data_root)
        print(f"Total images (from {args.dataset} dataset): {len(all_images)}")
    else:
        # Use train width's directory as the reference for file structure
        ref_path = args.data_root / f"w{args.train_width}{suffix}"
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference directory not found: {ref_path}")

        # Collect all image indices from the reference width, grouped by original split and digit
        ref_files = collect_files(ref_path)

        # Build a flat list of (original_split, digit, index) for all images
        # original_split is "train" or "test" from FMNIST (60k train + 10k test)
        all_images = []
        for key, paths in sorted(ref_files.items()):
            orig_split, digit = key.split("/")
            for p in paths:
                idx = get_image_index(p)
                all_images.append((orig_split, digit, idx))

        print(f"Total images: {len(all_images)}")

    # Shuffle all images with fixed seed, then partition into non-overlapping groups
    rng = random.Random(args.seed)
    rng.shuffle(all_images)

    # Allocate:
    #   - Each test width (including train width's test): test_size images
    #   - Train width val: val_size images
    #   - Train width train: all remaining images
    test_per_width = args.test_size
    val_size = args.val_size
    needed_for_test = n_widths * test_per_width  # all widths get a test set
    needed_for_val = val_size

    total_needed = needed_for_test + needed_for_val
    if total_needed > len(all_images):
        raise ValueError(f"Need {total_needed} images for test+val but only have {len(all_images)}")

    train_size = len(all_images) - total_needed
    print(f"Train width w{args.train_width}: {train_size} train / {val_size} val / {test_per_width} test")
    print(f"Test widths: {test_per_width} test each")
    print(f"Total allocated: {total_needed + train_size} / {len(all_images)}")

    # Partition
    cursor = 0

    # Train width: train set (all remaining after test+val)
    train_images = all_images[cursor:cursor + train_size]
    cursor += train_size

    # Train width: val set
    val_images = all_images[cursor:cursor + val_size]
    cursor += val_size

    # Test sets for each width
    width_test_images = {}
    for w in all_widths:
        width_test_images[w] = all_images[cursor:cursor + test_per_width]
        cursor += test_per_width

    assert cursor == len(all_images)

    # Disjointness, asserted rather than assumed: no image may serve two roles.
    seen = set(train_images) | set(val_images)
    assert len(seen) == len(train_images) + len(val_images), "train/val overlap"
    for w in all_widths:
        s = set(width_test_images[w])
        assert len(s) == test_per_width, f"w{w}: duplicate images within its own test set"
        assert not (s & seen), f"w{w}: test images overlap train/val or another width's test set"
        seen |= s
    print(f"Disjointness verified across {len(all_widths)} test sets + train/val ({len(seen):,} images)")

    prefix = "fmnist" if args.dataset == "fmnist" else "mnist"

    # Now write split JSONs for each width
    for w in all_widths:
        w_path = args.data_root / f"w{w}{suffix}"
        if not w_path.exists():
            if not args.create_dirs:
                print(f"[WARN] {w_path} does not exist, skipping w={w}")
                continue
            for split_dir in (f"{prefix}_png_train", f"{prefix}_png_test"):
                for digit in range(10):
                    (w_path / split_dir / str(digit)).mkdir(parents=True, exist_ok=True)
            print(f"  created empty {w_path}")

        split_data = {}

        if w == args.train_width:
            # Has train, val, and test
            split_data["train"] = _resolve_paths(train_images, w_path, args.dataset)
            split_data["val"] = _resolve_paths(val_images, w_path, args.dataset)
            split_data["test"] = _resolve_paths(width_test_images[w], w_path, args.dataset)
        else:
            # Test-only (but still provide train/val as empty for compatibility)
            split_data["train"] = {"path": [], "label": []}
            split_data["val"] = {"path": [], "label": []}
            split_data["test"] = _resolve_paths(width_test_images[w], w_path, args.dataset)

        save_path = w_path / args.split_name
        with open(save_path, "w") as f:
            json.dump(split_data, f)

        n_train = len(split_data["train"]["path"])
        n_val = len(split_data["val"]["path"])
        n_test = len(split_data["test"]["path"])
        print(f"  w={w:3d}: {n_train:6d} train / {n_val:5d} val / {n_test:5d} test -> {save_path}")


def _resolve_paths(image_list: list[tuple], w_path: Path, dataset: str) -> dict:
    """Convert (orig_split, digit, index) tuples to absolute paths and labels."""
    prefix = "fmnist" if dataset == "fmnist" else "mnist"
    paths = []
    labels = []
    for orig_split, digit, idx in image_list:
        split_dir = f"{prefix}_png_{orig_split}"
        p = w_path / split_dir / digit / f"{idx}.pth"
        paths.append(str(p.resolve()))
        labels.append(digit)
    return {"path": paths, "label": labels}


if __name__ == "__main__":
    main()
