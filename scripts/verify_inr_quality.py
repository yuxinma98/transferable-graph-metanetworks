#!/usr/bin/env python3
"""
Verify INR reconstruction quality across widths.
Computes MSE, PSNR and failure rate for generated INRs.

Usage:
    python scripts/verify_inr_quality.py --widths 16 24 32 48 64 80 96
    python scripts/verify_inr_quality.py --widths 16 32 --init-type mup
    python scripts/verify_inr_quality.py --dataset fmnist --widths 16 32 --init-type sp_d3
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

# Make src/ importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.coords import make_coord_grid

#: MSE above this counts as a failed fit.
FAIL_MSE_THRESHOLD = 0.01


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--widths", type=int, nargs="+", required=True)
    p.add_argument("--dataset", choices=["mnist", "fmnist"], default="mnist")
    p.add_argument(
        "--init-type",
        default="sp",
        help="Width-directory suffix: sp, mup, mup24, sp_d3, mup_d3, ...",
    )
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--data-root", type=Path, default=None)
    return p.parse_args()


def reconstruct_from_state_dict(state_dict, coords):
    """Reconstruct image from INR state dict."""
    num_layers = len(state_dict) // 2
    x = coords  # [784, 2]

    # Hidden layers with ReLU
    for layer_idx in range(num_layers - 1):
        W = state_dict[f"layers.{layer_idx}.weight"].T  # [in, out]
        b = state_dict[f"layers.{layer_idx}.bias"]
        x = torch.relu(x @ W + b)

    # Output layer with sigmoid
    W_out = state_dict[f"layers.{num_layers - 1}.weight"].T
    b_out = state_dict[f"layers.{num_layers - 1}.bias"]
    pred = torch.sigmoid(x @ W_out + b_out).squeeze()

    return pred  # [784]


def compute_psnr(mse):
    """Convert MSE to PSNR (dB)."""
    if mse == 0:
        return 100.0
    return -10 * np.log10(mse)


def build_image_map(dataset_name: str, data_root: Path) -> dict:
    """Map (split_id, label, intra_class_idx) → flattened ground-truth image.

    The n-th .pth file under {prefix}_png_{split}/{label}/ was fitted to the n-th
    occurrence of that label in that split, so counting occurrences in dataset
    order recovers the file → image correspondence used by generate_inrs.py.
    """
    ds_cls = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
    image_map = {}
    counters = {(split, label): 0 for split in (0, 1) for label in range(10)}
    for split_id, is_train in ((0, True), (1, False)):
        ds = ds_cls(data_root, train=is_train, download=False)
        for img, label in ds:
            idx = counters[(split_id, label)]
            counters[(split_id, label)] += 1
            image_map[(split_id, label, idx)] = transforms.ToTensor()(img).flatten()
    return image_map


def main():
    args = parse_args()

    if args.data_root is None:
        args.data_root = Path(os.environ.get("ANYDIM_DATA_ROOT", "data"))

    prefix = "fmnist" if args.dataset == "fmnist" else "mnist"
    image_map = build_image_map(args.dataset, args.data_root)
    coords = make_coord_grid()  # [784, 2]

    print(f"\n{'=' * 60}")
    print(f"INR Reconstruction Quality Verification ({args.dataset}, {args.init_type})")
    print(f"{'=' * 60}\n")

    results = []

    for width in args.widths:
        inr_dir = args.data_root / f"{prefix}_inrs" / f"w{width}_{args.init_type}"
        if not inr_dir.exists():
            print(f"⚠️  w{width}: Directory not found: {inr_dir}")
            continue

        train_files = sorted(glob.glob(str(inr_dir / f"{prefix}_png_train" / "*" / "*.pth")))
        test_files = sorted(glob.glob(str(inr_dir / f"{prefix}_png_test" / "*" / "*.pth")))

        all_files = train_files + test_files
        if len(all_files) == 0:
            print(f"⚠️  w{width}: No INR files found")
            continue

        # Sample uniformly
        if len(all_files) > args.num_samples:
            indices = np.linspace(0, len(all_files) - 1, args.num_samples, dtype=int)
            sample_files = [all_files[i] for i in indices]
        else:
            sample_files = all_files

        mse_list = []
        psnr_list = []

        for inr_path in tqdm(sample_files, desc=f"w{width}", leave=False):
            # Parse path: .../mnist_png_train/3/000042.pth
            parts = Path(inr_path).parts
            split = 0 if "train" in parts[-3] else 1
            label = int(parts[-2])
            idx = int(Path(inr_path).stem)

            gt_img = image_map.get((split, label, idx))
            if gt_img is None:
                continue  # no matching ground-truth image

            state_dict = torch.load(inr_path, map_location="cpu")
            with torch.no_grad():
                pred_img = reconstruct_from_state_dict(state_dict, coords)

            mse = F.mse_loss(pred_img, gt_img).item()
            mse_list.append(mse)
            psnr_list.append(compute_psnr(mse))

        if len(mse_list) == 0:
            print(f"⚠️  w{width}: No valid samples")
            continue

        mse_arr = np.array(mse_list)
        psnr_arr = np.array(psnr_list)

        fail_rate = (mse_arr > FAIL_MSE_THRESHOLD).mean()

        # Stats
        mse_mean = mse_arr.mean()
        mse_median = np.median(mse_arr)
        mse_std = mse_arr.std()

        psnr_mean = psnr_arr.mean()
        psnr_median = np.median(psnr_arr)
        psnr_std = psnr_arr.std()

        results.append(
            {
                "width": width,
                "mse_mean": mse_mean,
                "mse_median": mse_median,
                "mse_std": mse_std,
                "psnr_mean": psnr_mean,
                "psnr_median": psnr_median,
                "psnr_std": psnr_std,
                "fail_rate": fail_rate,
                "n_samples": len(mse_list),
            }
        )

        print(
            f"w{width:3d}: PSNR={psnr_mean:.1f}±{psnr_std:.1f} dB "
            f"(median={psnr_median:.1f}), Fail={fail_rate:.1%}"
        )

    # Summary table
    print(f"\n{'=' * 60}")
    print("Summary Table")
    print(f"{'=' * 60}\n")
    print(
        f"{'Width':>6} {'MSE mean':>10} {'MSE median':>11} {'PSNR mean':>10} "
        f"{'PSNR med':>9} {'PSNR std':>9} {'Fail%':>6}"
    )
    print("-" * 72)
    for r in results:
        print(
            f"{r['width']:6d} {r['mse_mean']:10.6f} {r['mse_median']:11.6f} "
            f"{r['psnr_mean']:9.1f} dB {r['psnr_median']:8.1f} dB "
            f"{r['psnr_std']:8.1f} {r['fail_rate']*100:5.1f}%"
        )

    print()


if __name__ == "__main__":
    main()
