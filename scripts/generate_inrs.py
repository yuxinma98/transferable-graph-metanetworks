"""
Generate INR datasets for MNIST or Fashion-MNIST.

Fits ReLU INRs to all 70K images (60K train + 10K test) and saves
individual .pth files in a directory structure compatible with ScaleGMN:

    out_dir/w{W}{suffix}/{prefix}_png_train/{digit}/{index:06d}.pth
    out_dir/w{W}{suffix}/{prefix}_png_test/{digit}/{index:06d}.pth

where suffix is `_sp` for standard parameterization and `_mup` for muP.

Each .pth file is a state dict with keys:
    layers.{i}.weight  [out_dim, in_dim]
    layers.{i}.bias    [out_dim]

Usage
-----
# Standard parameterization, one dataset per width:
python scripts/generate_inrs.py --widths 16 24 32 48 64 80 96

# muP (base_width=32), saves to w{W}_mup/:
python scripts/generate_inrs.py --widths 16 24 32 48 64 80 96 --mup --base-width 32

# Fashion-MNIST:
python scripts/generate_inrs.py --dataset fmnist --widths 16 24 32 48

# Only the INRs listed in a split JSON (for large widths, where fitting all 70k
# images would be wasteful — only the test set is ever evaluated):
python scripts/generate_inrs.py --widths 512 \
    --only-from-split $ANYDIM_DATA_ROOT/mnist_inrs/w512_sp/mnist_sizegen_splits_v4.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# Make src/ importable when running as a script
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.coords import make_coord_grid
from data.fit_inrs import BatchINRFitter, MuBatchINRFitter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate INR datasets for MNIST/FMNIST")
    p.add_argument(
        "--dataset",
        choices=["mnist", "fmnist"],
        default="mnist",
        help="Which image dataset to fit INRs to.",
    )
    p.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[32],
        help="Hidden width(s) to generate; one dataset directory per width.",
    )
    p.add_argument("--mup", action="store_true", help="Use muP instead of standard parameterization")
    p.add_argument(
        "--base-width",
        type=int,
        default=32,
        help="muP base width (ignored if --mup is not set)",
    )
    p.add_argument("--output-suffix", type=str, default=None, help="Override directory suffix (e.g. 'mup24' → w{N}_mup24)")
    p.add_argument("--num-hidden-layers", type=int, default=2, help="Number of hidden layers (default 2: [2,w,w,1])")
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=512, help="INRs fitted in parallel")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: auto from dataset name)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (affects init)")
    p.add_argument(
        "--only-from-split",
        type=Path,
        default=None,
        help="Fit only the INRs listed in this split JSON instead of all 70k images. "
             "Filenames are unaffected: they still come from a full pass over the dataset, "
             "so the same file name refers to the same source image at every width.",
    )
    p.add_argument(
        "--only-split-keys",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Which keys of --only-from-split to fit (default: all three)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose .pth file is already on disk (safe to resume an interrupted fit)",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("ANYDIM_DATA_ROOT", "data")),
        help="Download root for the source image dataset",
    )
    return p.parse_args()


def load_images(dataset_name: str, data_root: Path):
    """Load all 70K images. Returns (images [N,28,28], labels [N], split_ids [N])."""
    ds_cls = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
    train_ds = ds_cls(root=data_root, train=True, download=True, transform=transforms.ToTensor())
    test_ds = ds_cls(root=data_root, train=False, download=True, transform=transforms.ToTensor())

    train_images = train_ds.data.float() / 255.0  # [60000, 28, 28]
    test_images = test_ds.data.float() / 255.0    # [10000, 28, 28]
    train_labels = train_ds.targets                # [60000]
    test_labels = test_ds.targets                  # [10000]

    # split_id: 0 = train, 1 = test
    train_split = torch.zeros(len(train_images), dtype=torch.long)
    test_split = torch.ones(len(test_images), dtype=torch.long)

    all_images = torch.cat([train_images, test_images], dim=0)
    all_labels = torch.cat([train_labels, test_labels], dim=0)
    all_splits = torch.cat([train_split, test_split], dim=0)
    return all_images, all_labels, all_splits


def output_suffix(args: argparse.Namespace) -> str:
    """Directory suffix encoding parameterization and depth, e.g. '_sp', '_mup_d3'."""
    depth_tag = f"_d{args.num_hidden_layers}" if args.num_hidden_layers != 2 else ""
    base = args.output_suffix or ("mup" if args.mup else "sp")
    return f"_{base}{depth_tag}"


def canonical_file_ids(
    all_labels: torch.Tensor, all_splits: torch.Tensor, prefix: str
) -> list[tuple[str, str, str]]:
    """Output file identity of every image, as (png split dir, digit, stem).

    Derived from a full pass over the dataset in its fixed order, so a given
    file name always refers to the same source image regardless of which subset
    is actually fitted.
    """
    split_names = {0: f"{prefix}_png_train", 1: f"{prefix}_png_test"}
    counters = {(split, digit): 0 for split in (0, 1) for digit in range(10)}
    file_ids = []
    for label, split_id in zip(all_labels.tolist(), all_splits.tolist()):
        idx = counters[(split_id, label)]
        counters[(split_id, label)] += 1
        file_ids.append((split_names[split_id], str(label), f"{idx:06d}"))
    return file_ids


def load_split_filter(split_path: Path, keys: list[str]) -> set[tuple[str, str, str]]:
    """Image identities listed under `keys` of a split JSON."""
    with open(split_path) as f:
        split = json.load(f)
    wanted = set()
    for key in keys:
        for p in split.get(key, {}).get("path", []):
            parts = Path(p).parts
            wanted.add((parts[-3], parts[-2], Path(p).stem))
    return wanted


def generate_for_width(
    width: int,
    all_images: torch.Tensor,
    all_labels: torch.Tensor,
    all_splits: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    dims = [2] + [width] * args.num_hidden_layers + [1]
    print(f"\n=== Width {width}: architecture {dims} ===")

    out_root = args.out_dir / f"w{width}{output_suffix(args)}"
    coords = make_coord_grid().to(device)  # [784, 2]

    prefix = "fmnist" if args.dataset == "fmnist" else "mnist"
    for split_name in (f"{prefix}_png_train", f"{prefix}_png_test"):
        for digit in range(10):
            (out_root / split_name / str(digit)).mkdir(parents=True, exist_ok=True)

    file_ids = canonical_file_ids(all_labels, all_splits, prefix)
    selected = list(range(len(all_images)))

    if args.only_from_split is not None:
        if args.only_from_split.parent.name != out_root.name:
            print(f"  [WARN] split lives in {args.only_from_split.parent.name} but fitting into "
                  f"{out_root.name}; only the image identities are read, so check this is intended.")
        wanted = load_split_filter(args.only_from_split, args.only_split_keys)
        selected = [i for i in selected if file_ids[i] in wanted]
        if len(selected) != len(wanted):
            missing = wanted - {file_ids[i] for i in selected}
            raise ValueError(
                f"{len(missing)} image(s) in {args.only_from_split} do not exist in the "
                f"{prefix.upper()} dataset (e.g. {sorted(missing)[:3]})"
            )
        print(f"  Subset from {args.only_from_split.name} "
              f"({'+'.join(args.only_split_keys)}): {len(selected):,} of {len(all_images):,} images")

    if args.skip_existing:
        before = len(selected)
        selected = [i for i in selected if not (out_root / Path(*file_ids[i][:2]) / f"{file_ids[i][2]}.pth").exists()]
        print(f"  Skipping {before - len(selected):,} already-fitted INRs")

    if not selected:
        print("  Nothing to fit.")
        return

    for start in tqdm(range(0, len(selected), args.batch_size), desc=f"  w={width}"):
        batch_idx = selected[start:start + args.batch_size]
        bs = len(batch_idx)
        idx_t = torch.tensor(batch_idx)

        pixels = all_images[idx_t].reshape(bs, -1).to(device)  # [bs, 784]

        # Build fitter
        if args.mup:
            fitter = MuBatchINRFitter(bs, dims=dims, base_width=args.base_width).to(device)
            optimizer = fitter.get_mup_optimizer(args.lr)
        else:
            fitter = BatchINRFitter(bs, dims=dims).to(device)
            optimizer = torch.optim.Adam(fitter.parameters(), lr=args.lr)

        # Training loop
        for _ in range(args.num_steps):
            pred = fitter(coords)
            loss = F.mse_loss(pred, pixels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            state_dicts = fitter.get_state_dicts()

        for sd, i in zip(state_dicts, batch_idx):
            split_name, digit, stem = file_ids[i]
            torch.save(sd, out_root / split_name / digit / f"{stem}.pth")

        del fitter, optimizer
        torch.cuda.empty_cache()

    print(f"  Saved {len(selected):,} per-file INRs to {out_root}/")


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    if args.out_dir is None:
        data_root = Path(os.environ.get("ANYDIM_DATA_ROOT", "data"))
        subdir = "fmnist_inrs" if args.dataset == "fmnist" else "mnist_inrs"
        args.out_dir = data_root / subdir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_images, all_labels, all_splits = load_images(args.dataset, args.data_root)
    ds_label = "Fashion-MNIST" if args.dataset == "fmnist" else "MNIST"
    print(f"Loaded {len(all_images)} {ds_label} images")

    for width in args.widths:
        generate_for_width(width, all_images, all_labels, all_splits, args, device)

    print("\nDone. Run scripts/generate_sizegen_splits.py to create ScaleGMN-compatible JSON splits.")


if __name__ == "__main__":
    main()
