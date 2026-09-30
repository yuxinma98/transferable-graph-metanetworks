#!/usr/bin/env python3
"""
Sweep learning rates for INR generation at a given width.
Generates a small subset of INRs (500 samples) for each LR candidate,
evaluates reconstruction quality, and returns the best LR.

Supports both parameterizations and both image datasets:

    # SP, MNIST (default)
    python scripts/sweep_inr_lr.py --width 32

    # SP, Fashion-MNIST, extending the grid until the winner is interior
    python scripts/sweep_inr_lr.py --dataset fmnist --width 256 --auto-extend

    # muP with base_width=32 (only the base width needs sweeping; the muP
    # optimizer transfers that LR to every other width)
    python scripts/sweep_inr_lr.py --dataset fmnist --width 32 --mup --base-width 32
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from tqdm import tqdm

# Make src/ importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.coords import make_coord_grid
from data.fit_inrs import BatchINRFitter, MuBatchINRFitter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--dataset", choices=["mnist", "fmnist"], default="mnist")
    p.add_argument("--lr-candidates", type=float, nargs="+", default=[0.002, 0.005, 0.01, 0.02])
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--num-samples", type=int, default=500, help="Number of samples to fit for evaluation")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mup", action="store_true", help="Sweep the muP fitter instead of standard parameterization")
    p.add_argument("--base-width", type=int, default=32, help="muP base width (ignored without --mup)")
    p.add_argument("--output-file", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument(
        "--auto-extend",
        action="store_true",
        help="Keep extending the grid (halving below / doubling above) until the selected LR is "
             "an interior point, so the optimum is never reported at the edge of the search.",
    )
    p.add_argument("--max-extensions", type=int, default=5, help="Cap on --auto-extend steps")
    p.add_argument("--min-lr", type=float, default=1e-5, help="Lower bound for --auto-extend")
    p.add_argument("--max-lr", type=float, default=0.5, help="Upper bound for --auto-extend")
    p.add_argument(
        "--max-fail-rate",
        type=float,
        default=None,
        help="Opt-in alternative rule: pick the highest-PSNR LR among candidates whose failure "
             "rate is at or below this cap. Default (unset) minimizes failure rate, breaking "
             "ties on PSNR — that is the rule every INR dataset on disk was selected with, so "
             "leave this unset unless you intend to deviate. Note that at 500 samples the "
             "failure-rate estimate has an SE of ~0.45pp near 1%, so close rankings are noise.",
    )
    return p.parse_args()


def reconstruct_from_state_dict(state_dict, coords):
    """Reconstruct image from INR state dict."""
    num_layers = len(state_dict) // 2
    x = coords  # [784, 2]

    for layer_idx in range(num_layers - 1):
        W = state_dict[f"layers.{layer_idx}.weight"].T
        b = state_dict[f"layers.{layer_idx}.bias"]
        x = torch.relu(x @ W + b)

    W_out = state_dict[f"layers.{num_layers - 1}.weight"].T
    b_out = state_dict[f"layers.{num_layers - 1}.bias"]
    pred = torch.sigmoid(x @ W_out + b_out).squeeze()
    return pred


def compute_psnr(mse):
    if mse == 0:
        return 100.0
    return -10 * np.log10(mse)


def load_sample_images(args) -> torch.Tensor:
    """`--num-samples` images spread uniformly over the dataset's train split."""
    ds_cls = datasets.MNIST if args.dataset == "mnist" else datasets.FashionMNIST
    train_ds = ds_cls(args.data_root, train=True, download=False)
    indices = np.linspace(0, len(train_ds) - 1, args.num_samples, dtype=int)
    return train_ds.data[indices].float() / 255.0  # [num_samples, 28, 28]


def evaluate_lr(width, lr, args, device, images=None):
    """Generate subset of INRs with given LR and evaluate quality."""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if images is None:
        images = load_sample_images(args)
    coords = make_coord_grid().to(device)

    # Fit INRs in batches
    dims = [2, width, width, 1]
    mse_list = []

    for start in tqdm(range(0, args.num_samples, args.batch_size), desc=f"  lr={lr}", leave=False):
        end = min(start + args.batch_size, args.num_samples)
        bs = end - start

        pixels = images[start:end].reshape(bs, -1).to(device)

        # Fit — same fitter/optimizer pairing that generate_inrs.py uses
        if args.mup:
            fitter = MuBatchINRFitter(bs, dims=dims, base_width=args.base_width).to(device)
            optimizer = fitter.get_mup_optimizer(lr)
        else:
            fitter = BatchINRFitter(bs, dims=dims).to(device)
            optimizer = torch.optim.Adam(fitter.parameters(), lr=lr)

        for _ in range(args.num_steps):
            pred = fitter(coords)
            loss = F.mse_loss(pred, pixels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        with torch.no_grad():
            final_pred = fitter(coords)
            batch_mse = F.mse_loss(final_pred, pixels, reduction='none').mean(dim=1).cpu().numpy()
            mse_list.extend(batch_mse)

        del fitter, optimizer
        torch.cuda.empty_cache()

    mse_arr = np.array(mse_list)
    fail_rate = (mse_arr > 0.01).mean()
    psnr_list = [compute_psnr(m) for m in mse_arr]
    psnr_mean = np.mean(psnr_list)
    psnr_std = np.std(psnr_list)

    return {
        "lr": lr,
        "mse_mean": mse_arr.mean(),
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "fail_rate": fail_rate,
    }


def select_best(results: list[dict], max_fail_rate: float | None, verbose: bool = True) -> dict:
    """Pick one candidate. Default rule: lowest failure rate, ties broken on PSNR."""
    if max_fail_rate is not None:
        eligible = [r for r in results if r["fail_rate"] <= max_fail_rate]
        if eligible:
            if verbose:
                print(f"Selecting on PSNR among {len(eligible)}/{len(results)} candidates with "
                      f"fail rate <= {max_fail_rate:.1%}")
            return max(eligible, key=lambda r: r["psnr_mean"])
        if verbose:
            print(f"[WARN] no candidate meets the {max_fail_rate:.1%} failure cap; "
                  f"falling back to the lowest failure rate")
    return min(results, key=lambda r: (r["fail_rate"], -r["psnr_mean"]))


def main():
    args = parse_args()

    if args.data_root is None:
        _data_root = os.environ.get("ANYDIM_DATA_ROOT", "data")
        args.data_root = Path(_data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_tag = f"muP(base_width={args.base_width})" if args.mup else "SP"

    print(f"\n{'=' * 60}")
    print(f"LR Sweep for w{args.width} — {args.dataset}, {param_tag}")
    print(f"{'=' * 60}\n")
    print(f"Testing LRs: {args.lr_candidates}")
    print(f"Samples: {args.num_samples}, Steps: {args.num_steps}")
    print(f"Auto-extend: {args.auto_extend} (max {args.max_extensions} steps, "
          f"bounds [{args.min_lr}, {args.max_lr}])")
    print(f"Device: {device}\n")

    images = load_sample_images(args)
    evaluated: dict[float, dict] = {}
    grid = sorted(args.lr_candidates)

    for extension in range(args.max_extensions + 1):
        for lr in grid:
            if lr in evaluated:
                continue
            print(f"Testing lr={lr}...")
            evaluated[lr] = evaluate_lr(args.width, lr, args, device, images=images)
            r = evaluated[lr]
            print(f"  → PSNR={r['psnr_mean']:.1f}±{r['psnr_std']:.1f} dB, "
                  f"Fail={r['fail_rate']:.1%}\n")

        best = select_best([evaluated[lr] for lr in grid], args.max_fail_rate)
        if not args.auto_extend:
            break

        at_low, at_high = best["lr"] == grid[0], best["lr"] == grid[-1]
        if not (at_low or at_high):
            break
        if extension == args.max_extensions:
            print(f"[WARN] best LR {best['lr']} is still at the {'low' if at_low else 'high'} edge "
                  f"of the grid after {args.max_extensions} extensions — treat it as un-bracketed.")
            break
        nxt = grid[0] / 2 if at_low else grid[-1] * 2
        if nxt < args.min_lr or nxt > args.max_lr:
            print(f"[WARN] best LR {best['lr']} sits at the grid edge and the next candidate "
                  f"({nxt}) is outside [{args.min_lr}, {args.max_lr}] — stopping there.")
            break
        print(f"--- best lr={best['lr']} is at the {'low' if at_low else 'high'} edge; "
              f"extending the grid with {nxt} ---\n")
        grid = sorted(grid + [nxt])

    print(f"{'=' * 60}")
    print(f"Grid for w{args.width} ({len(grid)} candidates):")
    for lr in grid:
        r = evaluated[lr]
        mark = " <-- selected" if lr == best["lr"] else ""
        print(f"  lr={lr:<10g} PSNR={r['psnr_mean']:5.1f} dB  Fail={r['fail_rate']:6.1%}{mark}")
    interior = grid[0] < best["lr"] < grid[-1]
    print(f"Best LR for w{args.width}: {best['lr']}  ({'interior' if interior else 'AT GRID EDGE'})")
    print(f"  PSNR: {best['psnr_mean']:.1f} dB, Fail rate: {best['fail_rate']:.1%}")
    print(f"{'=' * 60}\n")

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "a") as f:
            f.write(f"{args.width} {best['lr']}\n")
        print(f"Appended to: {args.output_file}")

    return best['lr']


if __name__ == "__main__":
    main()
