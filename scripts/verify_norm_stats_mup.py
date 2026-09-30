"""
Verify that under muP training, ||w||_{V_n} = Theta(1) as width grows.

Theory predicts (Yang & Hu 2023):
  Under muP: ||W^(l)|| = Theta(sqrt(n_l / n_{l-1})) throughout training
    => ||w||_{V_n} = sum_l sqrt(n_{l-1}/n_l) * ||W^(l)|| = Theta(1)

This script uses MuBatchINRFitter (which wraps the `mup` package) for both
init measurement and training. Results should show ||w||_Vn ≈ constant across
all widths.

Usage:
    python scripts/verify_norm_stats_mup.py [--widths 16 24 32 48 64 96 128 192 256]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.coords import make_coord_grid
from data.fit_inrs import MuBatchINRFitter


def spectral_norm(W: torch.Tensor) -> float:
    return torch.linalg.svdvals(W)[0].item()


def compute_norms_from_sd(sd: dict, layer_dims: list[int]):
    """Compute per-layer scaled spectral norm from a state dict."""
    n_layers = len(layer_dims) - 1
    components = []
    for ell in range(n_layers):
        W = sd[f"layers.{ell}.weight"]
        n_prev = layer_dims[ell]
        n_curr = layer_dims[ell + 1]
        sv = spectral_norm(W)
        components.append(np.sqrt(n_prev / n_curr) * sv)
    return components, sum(components)


def load_mnist_subset(data_root: Path, num_images: int):
    ds = datasets.MNIST(root=data_root, train=True, download=True, transform=transforms.ToTensor())
    images = ds.data[:num_images].float() / 255.0
    return images


def measure_at_init(widths: list[int], num_samples: int = 500, base_width: int = 32):
    print("=" * 70)
    print("muP INITIALIZATION (no training)")
    print("=" * 70)
    print(f"Architecture: [2, w, w, 1], base_width={base_width}, num_samples={num_samples}")
    print()
    print(f"{'Width':>6} | {'||w||_Vn':>14} | {'L0':>8} | {'L1':>8} | {'L2':>8} | {'Last/Total':>10}")
    print("-" * 70)

    for w in widths:
        layer_dims = [2, w, w, 1]
        norms = []
        per_layer = [[] for _ in range(3)]

        fitter = MuBatchINRFitter(num_samples, dims=layer_dims, base_width=base_width)
        state_dicts = fitter.get_state_dicts()
        for sd in state_dicts:
            comps, total = compute_norms_from_sd(sd, layer_dims)
            norms.append(total)
            for i, c in enumerate(comps):
                per_layer[i].append(c)

        mean_n = np.mean(norms)
        std_n = np.std(norms)
        means = [np.mean(pl) for pl in per_layer]
        last_frac = means[-1] / mean_n
        print(f"{w:>6} | {mean_n:.2f} ± {std_n:.2f}{' ':>3}| {means[0]:.4f} | {means[1]:.4f} | {means[2]:.4f} | {last_frac:.3f}")


def measure_trained(
    widths: list[int],
    images: torch.Tensor,
    device: torch.device,
    num_steps: int = 1000,
    lr: float = 1e-2,
    batch_size: int = 50,
    base_width: int = 32,
):
    print()
    print("=" * 70)
    print(f"muP TRAINED INRs (ReLU, {num_steps} steps, lr={lr}, MuAdam)")
    print("=" * 70)
    print(f"Architecture: [2, w, w, 1], base_width={base_width}, num_samples={len(images)}")
    print()
    print(f"{'Width':>6} | {'||w||_Vn':>14} | {'L0':>8} | {'L1':>8} | {'L2':>8} | {'Last/Total':>10}")
    print("-" * 70)

    coords = make_coord_grid().to(device)  # [784, 2]
    total = len(images)

    results = {}
    for w in widths:
        layer_dims = [2, w, w, 1]
        norms = []
        per_layer = [[] for _ in range(3)]

        for start in tqdm(range(0, total, batch_size), desc=f"  w={w}", leave=False):
            end = min(start + batch_size, total)
            bs = end - start
            pixels = images[start:end].reshape(bs, -1).to(device)

            fitter = MuBatchINRFitter(bs, dims=layer_dims, base_width=base_width).to(device)
            optimizer = fitter.get_mup_optimizer(lr)

            for _ in range(num_steps):
                pred = fitter(coords)
                loss = F.mse_loss(pred, pixels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                state_dicts = fitter.get_state_dicts()

            for sd in state_dicts:
                comps, total_norm = compute_norms_from_sd(sd, layer_dims)
                norms.append(total_norm)
                for i, c in enumerate(comps):
                    per_layer[i].append(c)

            del fitter, optimizer
            torch.cuda.empty_cache()

        mean_n = np.mean(norms)
        std_n = np.std(norms)
        means = [np.mean(pl) for pl in per_layer]
        last_frac = means[-1] / mean_n
        results[w] = (mean_n, std_n, means)
        print(f"{w:>6} | {mean_n:.2f} ± {std_n:.2f}{' ':>3}| {means[0]:.4f} | {means[1]:.4f} | {means[2]:.4f} | {last_frac:.3f}")

    print()
    print("Scaling check (||w||_Vn — should be ~constant if Theta(1)):")
    for w in sorted(results.keys()):
        print(f"  width={w:3d}: ||w||_Vn = {results[w][0]:.4f} ± {results[w][1]:.4f}")

    print()
    print("Ratio check (||w||_Vn / sqrt(width) — should decrease if Theta(1)):")
    for w in sorted(results.keys()):
        ratio = results[w][0] / np.sqrt(w)
        print(f"  width={w:3d}: ||w||/sqrt(w) = {ratio:.4f}")

    return results


def measure_trained_from_disk(widths: list[int], data_root: Path, suffix: str, num_samples: int = 500):
    """Measure norms of muP INRs already fitted on disk (w{N}_{suffix}/)."""
    print()
    print("=" * 70)
    print(f"muP TRAINED INRs FROM DISK (w{{N}}_{suffix}/)")
    print("=" * 70)
    print(f"Data root: {data_root}")
    print()
    print(f"{'Width':>6} | {'||w||_Vn':>14} | {'L0':>8} | {'L1':>8} | {'L2':>8} | {'Last/Total':>10}")
    print("-" * 70)

    results = {}
    for w in widths:
        train_dir = data_root / f"w{w}_{suffix}" / "mnist_png_train"
        if not train_dir.exists():
            print(f"{w:>6} | data not found at {train_dir}, skipping")
            continue
        pth_files = sorted(train_dir.rglob("*.pth"))[:num_samples]
        if not pth_files:
            print(f"{w:>6} | no .pth files found, skipping")
            continue

        layer_dims = [2, w, w, 1]
        norms = []
        per_layer = [[] for _ in range(3)]
        for f in pth_files:
            sd = torch.load(f, weights_only=True, map_location="cpu")
            comps, total = compute_norms_from_sd(sd, layer_dims)
            norms.append(total)
            for i, c in enumerate(comps):
                per_layer[i].append(c)

        mean_n = np.mean(norms)
        std_n = np.std(norms)
        means = [np.mean(pl) for pl in per_layer]
        results[w] = (mean_n, std_n, means)
        print(f"{w:>6} | {mean_n:.2f} ± {std_n:.2f}{' ':>3}| {means[0]:.4f} | {means[1]:.4f} | {means[2]:.4f} | {means[-1] / mean_n:.3f}")

    print()
    print("Scaling check (||w||_Vn — should be ~constant if Theta(1)):")
    for w in sorted(results.keys()):
        print(f"  width={w:3d}: ||w||_Vn = {results[w][0]:.4f} ± {results[w][1]:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", type=int, nargs="+", default=[16, 24, 32, 48, 64, 96, 128, 192, 256])
    parser.add_argument("--num-samples", type=int, default=500, help="Number of INRs to fit per width")
    parser.add_argument("--num-steps", type=int, default=1000, help="Training steps per INR")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=50, help="INRs fitted in parallel")
    parser.add_argument("--base-width", type=int, default=32, help="muP base width")
    parser.add_argument("--init-only", action="store_true", help="Only measure at init (no training)")
    parser.add_argument(
        "--trained-from-disk",
        metavar="SUFFIX",
        default=None,
        help="Instead of re-fitting, measure INRs already on disk in w{N}_<SUFFIX>/ (e.g. mup24)",
    )
    args = parser.parse_args()

    data_root = os.environ.get("ANYDIM_DATA_ROOT", str(PROJECT_ROOT / "data"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    measure_at_init(args.widths, num_samples=args.num_samples, base_width=args.base_width)

    if args.init_only:
        return

    if args.trained_from_disk:
        measure_trained_from_disk(
            args.widths,
            Path(data_root) / "mnist_inrs",
            args.trained_from_disk,
            num_samples=args.num_samples,
        )
        return

    images = load_mnist_subset(Path(data_root), args.num_samples)
    print(f"\nLoaded {len(images)} MNIST images for fitting")
    measure_trained(
        args.widths, images, device, args.num_steps, args.lr, args.batch_size, args.base_width
    )


if __name__ == "__main__":
    main()
