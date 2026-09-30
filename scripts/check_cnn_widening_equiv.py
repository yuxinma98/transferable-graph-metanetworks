"""
Verify the channel-widening equivalence for the small-CNN-zoo architecture.

The CNN analog of ``scripts/check_widening_equiv.py``.  Given a base CNN
``[1] -> conv3x3/s2 -> [c] -> conv3x3/s2 -> [c] -> conv3x3/s2 -> [c] -> GAP -> dense -> [10]``
we widen the channel counts by ``k`` per hidden layer using

    W_up^(l)[(i,b), (j,a), :, :] = B^(l)_{ij}[b, a, :, :]
    b_up^(l)[(i,b)]              = b^(l)[i]

and check that the widened CNN computes the *same function* on random 32x32
inputs.  For the Kronecker families ``B_{ij}[b, a] = W[i, j] * P[b, a]``.

Claim: function preservation needs only the row condition
``sum_a B_{ij}[b, a] = W[i, j]`` (``P 1 = 1`` in the Kronecker case), with
``k_0 = k_L = 1``.

Proof sketch (induction over layers): a convolution is linear in its input
channels, so if the layer-``l-1`` feature maps of the widened net are the base
maps with channel ``j`` repeated ``k_{l-1}`` times, then for every copy ``b`` of
base channel ``i``

    sum_{(j,a)} B_{ij}[b,a] * x_j = sum_j (sum_a B_{ij}[b,a]) * x_j
                                  = sum_j W[i,j] * x_j,

i.e. the widened pre-activation map of ``(i, b)`` equals the base one of ``i``.
ReLU is elementwise, so the invariant carries to layer ``l``.  Global average
pooling is linear and per-channel, hence commutes with the duplication, and the
dense head is an MLP layer with ``k_L = 1``, so its output is unchanged. QED

Families covered (all satisfy the row condition):
  uniform        P = 1 1^T / k_in                          (pure duplication)
  row-stoch      random P with P 1 = 1
  doubly-stoch   also P^T 1 = 1                            (needs k_out == k_in)
  general-bidir  blockwise, non-Kronecker, row + column conditions
  general        blockwise, non-Kronecker, row condition only

Usage
-----
    python scripts/check_cnn_widening_equiv.py
    python scripts/check_cnn_widening_equiv.py --widths 16 32 --k 2 3 4 --n-trials 5
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.zoo_cnn import (  # noqa: E402
    FAMILIES, ZooCNN, check_family_conditions, layer_layout, state_dict_to_wb,
    widen_wb, zoo_cnn_forward,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--widths", type=int, nargs="+", default=[16, 32],
                   help="Base channel counts to test")
    p.add_argument("--k", type=int, nargs="+", default=[2, 3, 4],
                   help="Width multipliers to test")
    p.add_argument("--n-trials", type=int, default=3, help="Random CNNs per (width, k)")
    p.add_argument("--batch", type=int, default=8, help="Number of input images")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64",
                   help="float64 by default: the two blockwise families build blocks of "
                        "O(1) entries whose row sums cancel down to the O(0.1) base "
                        "weight, so float32 cancellation alone costs ~1e-4 of absolute "
                        "error there (the Kronecker families pass at ~1e-8 in float32).")
    return p.parse_args()


def main():
    args = parse_args()
    gen = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)

    print("Channel-widening equivalence for the zoo CNN")
    print(f"widths={args.widths}  k={args.k}  trials={args.n_trials}  "
          f"tol={args.tol:.0e}  dtype={args.dtype}")
    print()

    header = (f"{'width':>6} {'k':>3} {'family':>14} "
              f"{'fn err':>10} {'row err':>10} {'col err':>10} {'nonunif':>10}")
    print(header)
    print("-" * len(header))

    worst = {fam: 0.0 for fam in FAMILIES}
    worst_row = {fam: 0.0 for fam in FAMILIES}
    # widths must scale so that the doubly-stochastic family (k_out == k_in) applies
    for width in args.widths:
        for k in args.k:
            for trial in range(args.n_trials):
                net = ZooCNN(width=width).to(dtype)
                sd = net.export_state_dict()
                weights, biases = state_dict_to_wb(sd)
                mults = [1] + [k] * (len(weights) - 1) + [1]

                x = torch.randn(args.batch, 1, args.image_size, args.image_size,
                                generator=gen, dtype=dtype)
                y_base = zoo_cnn_forward(weights, biases, x)

                for fam in FAMILIES:
                    wide_w, wide_b = widen_wb(weights, biases, mults, fam, generator=gen)
                    y_wide = zoo_cnn_forward(wide_w, wide_b, x)
                    fn_err = (y_base - y_wide).abs().max().item()
                    row_err, col_err, nonunif = check_family_conditions(
                        weights, wide_w, mults)
                    worst[fam] = max(worst[fam], fn_err)
                    worst_row[fam] = max(worst_row[fam], row_err)
                    if trial == 0:
                        print(f"{width:>6} {k:>3} {fam:>14} "
                              f"{fn_err:>10.2e} {row_err:>10.2e} {col_err:>10.2e} "
                              f"{nonunif:>10.2e}")

                    # sanity: the widened net really has the widened layout
                    wide_layout = [wide_w[0].shape[1]] + [b.shape[0] for b in wide_b]
                    expected = [n * m for n, m in zip(layer_layout(width), mults)]
                    assert wide_layout == expected, (wide_layout, expected)

    print("-" * len(header))
    print("\nSummary — max |f_base - f_wide| over all trials:")
    ok = True
    for fam in FAMILIES:
        passed = worst[fam] < args.tol
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FAIL':>4}  {fam:<14} "
              f"fn err={worst[fam]:.2e}  (row cond err={worst_row[fam]:.2e})")

    print()
    if ok:
        print("[OK] Every widening family preserves the represented function.")
    else:
        print("[FAIL] Some family broke functional equivalence.")
        sys.exit(1)


if __name__ == "__main__":
    main()
