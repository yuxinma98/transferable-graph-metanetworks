"""
Verify the cross-dimensional widening equivalence on real SIREN INR files.

Given a base MLP with weights W^(l) in R^{n_l x n_{l-1}} and biases b_l in R^{n_l},
construct a widened MLP via:

    W^(l)_up = W^(l) ⊗ P^(l)   in R^{N_l x N_{l-1}},   N_l = k_l * n_l
    b_l_up   = b_l ⊗ 1_{k_l}   in R^{N_l}

where P^(l) in R^{k_l x k_{l-1}} satisfies P^(l) 1 = 1 (rows sum to 1),
and k_0 = k_L = 1 (input/output dims unchanged).

Claim: both MLPs compute the same function.

Proof sketch (by induction): if x^{l-1}_up = x^{l-1} ⊗ 1_{k_{l-1}}, then
  W^(l)_up x^{l-1}_up + b_l_up
    = (W^(l) ⊗ P^(l))(x^{l-1} ⊗ 1_{k_{l-1}}) + b_l ⊗ 1_{k_l}
    = W^(l)x^{l-1} ⊗ P^(l)1_{k_{l-1}} + b_l ⊗ 1_{k_l}
    = (W^(l)x^{l-1} + b_l) ⊗ 1_{k_l}              [since P^(l)1 = 1]
so activation (elementwise) preserves the structure: x^{l}_up = x^{l} ⊗ 1_{k_l}.
At l=L, k_L=1, so x^{L}_up = x^{L}. QED.

Usage
-----
    python scripts/check_widening_equiv.py
    python scripts/check_widening_equiv.py --data-dir data/mnist_inrs/w32 --n-inrs 20 --n-trials 5
"""

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.coords import make_coord_grid  # noqa: E402

OMEGA_0 = 30.0


# ---------------------------------------------------------------------------
# MLP forward passes
# ---------------------------------------------------------------------------

def _n_layers(state_dict: dict) -> int:
    return max(int(k.split(".")[1]) for k in state_dict if "weight" in k) + 1


def forward_siren(state_dict: dict, coords: torch.Tensor) -> torch.Tensor:
    """Evaluate a SIREN MLP on coords.

    Args:
        state_dict: keys 'layers.{i}.weight' [out, in], 'layers.{i}.bias' [out]
        coords: [P, n_input]
    Returns:
        [P, n_output]
    """
    L = _n_layers(state_dict)
    x = coords
    for i in range(L):
        W = state_dict[f"layers.{i}.weight"]  # [out, in]
        b = state_dict[f"layers.{i}.bias"]    # [out]
        x = x @ W.T + b
        if i < L - 1:
            x = torch.sin(OMEGA_0 * x)
        else:
            x = torch.sigmoid(x)
    return x


def widen_state_dict(
    state_dict: dict,
    multipliers: list[int],
    Ps: list[torch.Tensor],
) -> dict:
    """Construct the widened state dict via Kronecker products.

    Args:
        state_dict: base MLP state dict
        multipliers: [k_0, k_1, ..., k_L] with k_0 = k_L = 1
        Ps: list of L matrices P^(l) in R^{k_l x k_{l-1}}, rows sum to 1
    Returns:
        widened state dict compatible with forward_siren
    """
    L = _n_layers(state_dict)
    assert len(multipliers) == L + 1, f"Need {L+1} multipliers for {L} layers"
    assert multipliers[0] == 1 and multipliers[-1] == 1, "k_0 and k_L must be 1"
    assert len(Ps) == L, f"Need {L} P matrices for {L} layers"

    widened = {}
    for i in range(L):
        W = state_dict[f"layers.{i}.weight"]  # [n_out, n_in]
        b = state_dict[f"layers.{i}.bias"]    # [n_out]
        P = Ps[i]                              # [k_out, k_in]

        # W_up = W ⊗ P  in R^{N_out x N_in},  N_out = k_out * n_out
        W_up = torch.kron(W.contiguous(), P)

        # b_up = b ⊗ 1_{k_out}  in R^{N_out}
        k_out = multipliers[i + 1]
        b_up = torch.kron(b, torch.ones(k_out, dtype=b.dtype))

        widened[f"layers.{i}.weight"] = W_up
        widened[f"layers.{i}.bias"] = b_up

    return widened


def random_row_stochastic(k_out: int, k_in: int, generator: torch.Generator) -> torch.Tensor:
    """Return a random row-stochastic matrix in R^{k_out x k_in}: rows sum to 1."""
    raw = torch.rand(k_out, k_in, generator=generator).abs() + 1e-6
    return raw / raw.sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def check_file(
    pth_path: Path,
    n_trials: int,
    max_k: int,
    coords: torch.Tensor,
    rng: torch.Generator,
) -> dict:
    """Run widening-equivalence checks on one INR file.

    Returns a dict with 'max_abs_err' (over all trials and coordinates).
    """
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    L = _n_layers(state_dict)

    base_out = forward_siren(state_dict, coords)  # [P, 1]

    max_err = 0.0
    for _ in range(n_trials):
        # Sample random width multipliers k_1, ..., k_{L-1} in [1, max_k]
        ks = [1] + [int(torch.randint(1, max_k + 1, (), generator=rng).item()) for _ in range(L - 1)] + [1]

        # Build row-stochastic P^(l) for each layer
        Ps = [random_row_stochastic(ks[i + 1], ks[i], rng) for i in range(L)]

        wide_sd = widen_state_dict(state_dict, ks, Ps)
        wide_out = forward_siren(wide_sd, coords)

        err = (wide_out - base_out).abs().max().item()
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}


def find_pth_files(data_dir: Path, limit: int) -> list[Path]:
    """Glob for .pth INR files under data_dir."""
    files = sorted(data_dir.rglob("*.pth"))
    return files[:limit]


def main():
    parser = argparse.ArgumentParser(description="Check cross-dimensional widening equivalence")
    _default = os.environ.get("ANYDIM_DATA_ROOT", str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(_default) / "mnist_inrs" / "w32",
        help="Directory containing .pth INR files (default: $ANYDIM_DATA_ROOT/mnist_inrs/w32)",
    )
    parser.add_argument("--n-inrs",   type=int, default=10, help="Number of INR files to check")
    parser.add_argument("--n-trials", type=int, default=5,  help="Random P trials per INR")
    parser.add_argument("--max-k",    type=int, default=4,  help="Max width multiplier k_l")
    parser.add_argument("--n-coords", type=int, default=784, help="Number of query coordinates")
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--tol",      type=float, default=1e-5, help="Pass threshold for max |error|")
    args = parser.parse_args()

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    coords = make_coord_grid(28, 28)  # [784, 2]

    files = find_pth_files(args.data_dir, args.n_inrs)
    if not files:
        print(f"[ERROR] No .pth files found under {args.data_dir}")
        sys.exit(1)

    print(f"Checking {len(files)} INR files from {args.data_dir}")
    print(f"  {args.n_trials} random P trials each, max_k={args.max_k}, tol={args.tol}\n")

    all_errs = []
    for pth in files:
        result = check_file(pth, args.n_trials, args.max_k, coords, rng)
        err = result["max_abs_err"]
        all_errs.append(err)
        status = "PASS" if err < args.tol else "FAIL"
        print(f"  [{status}]  {pth.relative_to(args.data_dir)}  max_abs_err={err:.2e}")

    print()
    print(f"Summary over {len(files)} files x {args.n_trials} trials:")
    print(f"  max  |error| = {max(all_errs):.2e}")
    print(f"  mean |error| = {sum(all_errs)/len(all_errs):.2e}")
    n_fail = sum(e >= args.tol for e in all_errs)
    if n_fail == 0:
        print(f"  All PASSED (tol={args.tol})")
    else:
        print(f"  {n_fail}/{len(all_errs)} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
