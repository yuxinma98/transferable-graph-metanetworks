"""
Verify the theoretical predictions about scaled spectral norm from the paper (Section 3).

Theory predicts:
  Under PyTorch default init: W^(l)_{ij} ~ N(0, 1/n_{l-1})
    => ||W^(l)|| ≈ 1 + sqrt(n_l / n_{l-1})
    => ||w||_{V_n} = sum_l sqrt(n_{l-1}/n_l) * ||W^(l)|| ≈ sum_l (1 + sqrt(n_{l-1}/n_l))
    => dominated by last layer term when n_0, n_L are fixed, grows as Theta(sqrt(n))

  Under muP: ||W^(l)|| = Theta(sqrt(n_l / n_{l-1})) throughout training
    => ||w||_{V_n} = Theta(1)

This script:
  1. Verifies the norm scaling at initialization for different widths.
  2. Loads trained INRs from the data directory and computes their norms.
  3. Computes per-layer contributions to see which dominates.

Usage:
    python scripts/verify_norm_stats.py [--widths 16 24 32 40 48 64 80 96]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent


def spectral_norm(W: torch.Tensor) -> float:
    """Compute operator 2-norm (largest singular value)."""
    return torch.linalg.svdvals(W)[0].item()


def scaled_spectral_norm_components(state_dict: dict, layer_dims: list[int]):
    """Compute per-layer scaled spectral norm contributions.

    Returns:
        components: list of sqrt(n_{l-1}/n_l) * ||W^(l)|| for each layer
        total: sum of components (= ||w||_{V_n})
    """
    components = []
    weights = [v for k, v in state_dict.items() if "weight" in k]
    for ell, W in enumerate(weights):
        n_ell_minus_1 = layer_dims[ell]
        n_ell = layer_dims[ell + 1]
        sv = spectral_norm(W)
        scaled = np.sqrt(n_ell_minus_1 / n_ell) * sv
        components.append(scaled)
    return components, sum(components)


def verify_at_initialization(widths, din=2, dout=1, num_samples=100):
    """Generate random MLPs at PyTorch default init and compute norms."""
    print("=" * 70)
    print("VERIFICATION AT INITIALIZATION (PyTorch default: Kaiming uniform)")
    print("=" * 70)
    print(f"Architecture: [{din}, w, w, {dout}], num_samples={num_samples}")
    print()

    results = {}
    for w in widths:
        layer_dims = [din, w, w, dout]
        norms = []
        per_layer = [[] for _ in range(len(layer_dims) - 1)]

        for _ in range(num_samples):
            sd = {}
            for ell in range(len(layer_dims) - 1):
                n_in = layer_dims[ell]
                n_out = layer_dims[ell + 1]
                # PyTorch default: kaiming_uniform_ with a=sqrt(5)
                W = torch.empty(n_out, n_in)
                torch.nn.init.kaiming_uniform_(W, a=np.sqrt(5))
                sd[f"layers.{ell}.weight"] = W

            comps, total = scaled_spectral_norm_components(sd, layer_dims)
            norms.append(total)
            for i, c in enumerate(comps):
                per_layer[i].append(c)

        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        results[w] = (mean_norm, std_norm, [np.mean(pl) for pl in per_layer])

        layer_str = "  ".join(
            f"L{i}={np.mean(per_layer[i]):.3f}" for i in range(len(layer_dims) - 1)
        )
        print(f"  width={w:3d}: ||w||_Vn = {mean_norm:.4f} ± {std_norm:.4f}  [{layer_str}]")

    # Check scaling: if theory correct, ratio ||w||/sqrt(w) should be ~constant
    print()
    print("  Scaling check (||w||_Vn / sqrt(width)):")
    for w in widths:
        ratio = results[w][0] / np.sqrt(w)
        print(f"    width={w:3d}: ratio = {ratio:.4f}")

    return results


def verify_trained_inrs(widths, data_root, num_samples=200):
    """Load trained INRs and compute their scaled spectral norms."""
    print()
    print("=" * 70)
    print("VERIFICATION ON TRAINED INRs (ReLU, standard parameterization)")
    print("=" * 70)
    print(f"Data root: {data_root}")
    print()

    results = {}
    for w in widths:
        data_dir = Path(data_root) / f"w{w}_sp"
        if not data_dir.exists():
            print(f"  width={w}: data not found at {data_dir}, skipping")
            continue

        train_dir = data_dir / "mnist_png_train"
        if not train_dir.exists():
            print(f"  width={w}: no mnist_png_train/ in {data_dir}, skipping")
            continue

        # Collect .pth files
        pth_files = sorted(train_dir.rglob("*.pth"))[:num_samples]
        if not pth_files:
            print(f"  width={w}: no .pth files found, skipping")
            continue

        layer_dims = [2, w, w, 1]
        norms = []
        per_layer = [[] for _ in range(len(layer_dims) - 1)]

        for f in pth_files:
            sd = torch.load(f, weights_only=True, map_location="cpu")
            comps, total = scaled_spectral_norm_components(sd, layer_dims)
            norms.append(total)
            for i, c in enumerate(comps):
                per_layer[i].append(c)

        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        results[w] = (mean_norm, std_norm, [np.mean(pl) for pl in per_layer])

        layer_str = "  ".join(
            f"L{i}={np.mean(per_layer[i]):.3f}±{np.std(per_layer[i]):.3f}"
            for i in range(len(layer_dims) - 1)
        )
        print(f"  width={w:3d}: ||w||_Vn = {mean_norm:.4f} ± {std_norm:.4f}  [{layer_str}]")

    if results:
        print()
        print("  Scaling check (||w||_Vn / sqrt(width)):")
        for w in sorted(results.keys()):
            ratio = results[w][0] / np.sqrt(w)
            print(f"    width={w:3d}: ratio = {ratio:.4f}")

        # Also check which layer dominates
        print()
        print("  Last-layer fraction of total norm:")
        for w in sorted(results.keys()):
            last_layer_frac = results[w][2][-1] / results[w][0]
            print(f"    width={w:3d}: L_last / total = {last_layer_frac:.4f}")

    return results


def verify_dws_inrs(data_root, num_samples=500):
    """Load DWS MNIST-INRs (width=32) and compute their scaled spectral norms."""
    dws_dir = Path(data_root) / "mnist-inrs"
    if not dws_dir.exists():
        print("\n  DWS data not found, skipping.")
        return {}

    print()
    print("=" * 70)
    print("VERIFICATION ON DWS MNIST-INRs (width=32, original ScaleGMN data)")
    print("=" * 70)

    # Find MNIST training files
    pth_files = sorted(dws_dir.glob("mnist_png_training_*/checkpoints/model_final.pth"))[:num_samples]
    if not pth_files:
        print("  No DWS files found.")
        return {}

    layer_dims = [2, 32, 32, 1]
    norms = []
    per_layer = [[] for _ in range(3)]

    for f in pth_files:
        sd = torch.load(f, weights_only=False, map_location="cpu")
        comps, total = scaled_spectral_norm_components(sd, layer_dims)
        norms.append(total)
        for i, c in enumerate(comps):
            per_layer[i].append(c)

    mean_norm = np.mean(norms)
    std_norm = np.std(norms)

    layer_str = "  ".join(
        f"L{i}={np.mean(per_layer[i]):.3f}±{np.std(per_layer[i]):.3f}"
        for i in range(3)
    )
    print(f"  DWS w=32: ||w||_Vn = {mean_norm:.4f} ± {std_norm:.4f}  [{layer_str}]")
    print(f"  Last-layer fraction: {np.mean(per_layer[-1]) / mean_norm:.4f}")

    return {"dws_32": (mean_norm, std_norm, [np.mean(pl) for pl in per_layer])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--widths", type=int, nargs="+", default=[16, 24, 32, 40, 48, 64, 80, 96]
    )
    parser.add_argument("--num-samples", type=int, default=200)
    args = parser.parse_args()

    data_root = os.environ.get("ANYDIM_DATA_ROOT", str(PROJECT_ROOT / "data"))
    data_root_inrs = Path(data_root) / "mnist_inrs"

    # 1. Theory verification at initialization
    verify_at_initialization(args.widths, num_samples=args.num_samples)

    # 2. Trained INRs (our generated SIREN data)
    verify_trained_inrs(args.widths, data_root_inrs, num_samples=args.num_samples)

    # 3. DWS INRs
    verify_dws_inrs(data_root, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
