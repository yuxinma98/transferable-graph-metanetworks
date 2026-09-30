"""
Generate widened INRs from a base-width test set, using any of the five equivalence families.

Given a base INR with architecture [2, w, w, 1], construct widened INRs with architecture
[2, w*k, w*k, 1] that compute the EXACT same function.  Copies of base neuron j occupy
contiguous indices [j*k, (j+1)*k), matching `torch.kron` and the checkers in
`scripts/check_matrix_product_gmn.py`.

A widening is written blockwise: with W^(l) in R^{n_out x n_in}, the widened weight is built
from blocks B^(l)_{ij} in R^{k_l x k_{l-1}},

    W^(l)_up[i*k_l + a, j*k_{l-1} + b] = B^(l)_{ij}[a, b],
    b^(l)_up = b^(l) (x) 1_{k_l},

and the function is preserved iff every block's row sums equal the base weight,

    (row)  B^(l)_{ij} 1_{k_{l-1}} = W^(l)_{ij} 1_{k_l}            (eq:widened-general-...)

which is all the FORWARD matrix-product aggregate ever sees.  The BACKWARD aggregate uses
W^T, so it additionally needs the column condition

    (col)  1_{k_l}^T B^(l)_{ij} = (k_l / k_{l-1}) W^(l)_{ij} 1_{k_{l-1}}^T

The five families (see results/VERIFICATION_gmn_properties.md § Equivalence classes):

    family         B_{ij}                                              satisfies
    ------------------------------------------------------------------------------
    uniform        W_ij (1/k_in) 1 1^T           (eq:widened-duplication)   row + col
    row-stoch      W_ij P,  P 1 = 1                                        row
    doubly-stoch   W_ij P,  P 1 = 1 and P^T 1 = (k_out/k_in) 1             row + col
    general-bidir  independent random block per (i, j), row + col imposed   row + col
    general        independent random block per (i, j), row only imposed    row

`general` is the largest class: the most general widening that preserves the function.
Only the forward matrix-product models are invariant to it; the edgewise-MLP GMNs are
invariant to `uniform` alone.

Blocks are built in float64 and cast to float32 on write (the free blocks are O(1) randn
entries whose row sums cancel down to the O(1) base weight, so the construction is done in
double to keep that cancellation exact).

Usage:
    # uniform Kronecker widening (the original behaviour) -> w{N}_{init}_dup/
    python scripts/generate_duplicated_inrs.py --init-type mup24 --base-width 24 \
        --target-widths 48 96 144 192 240

    # most general function-preserving widening -> w{N}_{init}_gen/
    python scripts/generate_duplicated_inrs.py --init-type mup24 --base-width 24 \
        --target-widths 48 96 144 192 240 --family general

    # FMNIST size-gen v2: base width 32, and its split lives under a v2 filename
    python scripts/generate_duplicated_inrs.py --dataset fmnist --init-type mup32 \
        --base-width 32 --target-widths 64 128 192 256 320 --family general \
        --split-name fmnist_sizegen_splits_v2.json
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent

FAMILIES = ("uniform", "row-stoch", "doubly-stoch", "general-bidir", "general")

# Directory suffix per family. `uniform` keeps `_dup` for backwards compatibility with the
# w{48..240}_{sp,mup24}_dup/ datasets already on disk.
FAMILY_SUFFIX = {
    "uniform": "dup",
    "row-stoch": "rowst",
    "doubly-stoch": "dblst",
    "general-bidir": "genbd",
    "general": "gen",
}

DTYPE = torch.float64


def _n_layers(state_dict: dict) -> int:
    return max(int(key.split(".")[1]) for key in state_dict if "weight" in key) + 1


def _row_stochastic(k_out: int, k_in: int, gen: torch.Generator) -> torch.Tensor:
    """Random P in R^{k_out x k_in} with P 1 = 1 (entries may be negative)."""
    p = torch.randn(k_out, k_in, dtype=DTYPE, generator=gen)
    return p + (1.0 - p.sum(-1, keepdim=True)) / k_in


def _equal_row_col_sums(shape, c: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random square blocks with every row sum AND every column sum equal to `c`.

    `shape` is (..., k, k) and `c` broadcasts against the leading dims. Obtained by
    projecting a random matrix onto that affine subspace (J = 1 1^T):
        B = R - (1/k) R J - (1/k) J R + (1/k^2) J R J + (c/k) J.
    Entries may be negative and c = 0 is allowed, so zero base weights are covered too.
    """
    k = shape[-1]
    assert shape[-2] == k, "equal row/column sums require a square block"
    r = torch.randn(*shape, dtype=DTYPE, generator=gen)
    j = torch.ones(k, k, dtype=DTYPE)
    return r - r @ j / k - j @ r / k + j @ r @ j / k**2 + c[..., None, None] * j / k


def _blocks(W: torch.Tensor, k_out: int, k_in: int, family: str,
            gen: torch.Generator) -> torch.Tensor:
    """Blocks B_{ij} in R^{k_out x k_in} for one layer. `W` is [n_out, n_in], float64.

    At the boundary layers the constraints force the block and every family agrees:
    k_in == 1 makes (row) pin every entry to W_ij, and k_out == 1 with (col) imposed
    pins the single row to W_ij 1^T / k_in.
    """
    n_out, n_in = W.shape
    uniform = W[:, :, None, None] * torch.ones(k_out, k_in, dtype=DTYPE) / k_in

    if family == "uniform" or k_in == 1:
        return uniform

    if family == "general":
        # free apart from (row): all row sums == W_ij
        b = torch.randn(n_out, n_in, k_out, k_in, dtype=DTYPE, generator=gen)
        return b + (W[:, :, None, None] - b.sum(-1, keepdim=True)) / k_in

    if family == "general-bidir":
        # free apart from (row) and (col) together. Squareness is what leaves any freedom:
        # for k_in != k_out (incl. the k_out == 1 output layer) the two conditions force
        # the uniform block.
        if k_in == k_out:
            return _equal_row_col_sums((n_out, n_in, k_out, k_in), W, gen)
        return uniform

    if family == "row-stoch":
        # W (x) P with P 1 = 1. The k_out == 1 output layer still has freedom here (a row
        # vector summing to 1), which is what keeps this family non-uniform end to end.
        return W[:, :, None, None] * _row_stochastic(k_out, k_in, gen)

    if family == "doubly-stoch":
        if k_out == 1:
            return uniform
        assert k_out == k_in, "doubly-stochastic widening needs k_l == k_{l-1}"
        p = _equal_row_col_sums((k_out, k_in), torch.ones((), dtype=DTYPE), gen)
        return W[:, :, None, None] * p

    raise ValueError(f"unknown family: {family}")


def widen_state_dict(state_dict: dict, k: int, family: str,
                     gen: torch.Generator) -> dict:
    """Widen a [2, w, ..., w, 1] INR to [2, w*k, ..., w*k, 1] with the given family."""
    L = _n_layers(state_dict)
    ks = [1] + [k] * (L - 1) + [1]          # k_0 = k_L = 1

    widened = {}
    for i in range(L):
        W = state_dict[f"layers.{i}.weight"].to(DTYPE)   # [n_out, n_in]
        b = state_dict[f"layers.{i}.bias"].to(DTYPE)     # [n_out]
        n_out, n_in = W.shape
        k_in, k_out = ks[i], ks[i + 1]

        blocks = _blocks(W, k_out, k_in, family, gen)     # [n_out, n_in, k_out, k_in]
        # [i, a, j, b] -> [n_out * k_out, n_in * k_in]
        W_up = blocks.permute(0, 2, 1, 3).reshape(n_out * k_out, n_in * k_in)

        widened[f"layers.{i}.weight"] = W_up.to(torch.float32)
        widened[f"layers.{i}.bias"] = torch.kron(
            b, torch.ones(k_out, dtype=DTYPE)).to(torch.float32)

    return widened


# ---------------------------------------------------------------------------
# Verification: is the widening actually function-preserving, and non-uniform?
# ---------------------------------------------------------------------------

def block_condition_errors(state_dict: dict, widened: dict, k: int):
    """(row_err, col_err, nonuniformity) as max abs deviations over all layers."""
    L = _n_layers(state_dict)
    ks = [1] + [k] * (L - 1) + [1]
    row_err = col_err = nonuniformity = 0.0
    for i in range(L):
        W = state_dict[f"layers.{i}.weight"].to(DTYPE)
        n_out, n_in = W.shape
        k_in, k_out = ks[i], ks[i + 1]
        blocks = widened[f"layers.{i}.weight"].to(DTYPE) \
            .reshape(n_out, k_out, n_in, k_in).permute(0, 2, 1, 3)
        row_err = max(row_err, (blocks.sum(-1) - W[:, :, None]).abs().max().item())
        col_err = max(col_err,
                      (blocks.sum(-2) - (k_out / k_in) * W[:, :, None]).abs().max().item())
        uniform = W[:, :, None, None] * torch.ones(k_out, k_in, dtype=DTYPE) / k_in
        nonuniformity = max(nonuniformity, (blocks - uniform).abs().max().item())
    return row_err, col_err, nonuniformity


def forward_inr(state_dict: dict, coords: torch.Tensor):
    """(logits, pixels) of the fitted ReLU INR. `BatchINRFitter` sigmoids the output layer,
    so `pixels` is the represented function and `logits` is the stricter pre-activation
    quantity (O(100) here, since fitting a binarized-looking image saturates the sigmoid)."""
    L = _n_layers(state_dict)
    h = coords.to(DTYPE)
    for i in range(L):
        h = h @ state_dict[f"layers.{i}.weight"].to(DTYPE).T \
            + state_dict[f"layers.{i}.bias"].to(DTYPE)
        if i < L - 1:
            h = torch.relu(h)
    return h, torch.sigmoid(h)


def main():
    # Imported here, not at module scope: `eval_sizegen_duplicated.py` imports this module
    # for FAMILY_SUFFIX after chdir'ing into src/scalegmn/, where `src` resolves to
    # ScaleGMN's own package and `src.data.coords` does not exist.
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.coords import make_coord_grid

    parser = argparse.ArgumentParser()
    parser.add_argument("--init-type", type=str, required=True,
                        help="Suffix for the source width dirs (e.g. sp, mup24, sp_d3)")
    parser.add_argument("--target-widths", type=int, nargs="+", default=[32, 48, 64, 80, 96])
    parser.add_argument("--base-width", type=int, default=16)
    parser.add_argument("--dataset", choices=["mnist", "fmnist"], default="mnist")
    parser.add_argument("--split-name", type=str, default=None,
                        help="Split JSON to read from the base-width dir and write into each "
                             "widened dir (default: {dataset}_sizegen_splits.json; FMNIST "
                             "size-gen v2 uses fmnist_sizegen_splits_v2.json)")
    parser.add_argument("--family", choices=FAMILIES, default="uniform",
                        help="Widening equivalence class (default: uniform Kronecker)")
    parser.add_argument("--out-suffix", type=str, default=None,
                        help="Override the output dir suffix (default: per-family, "
                             "uniform->dup, general->gen, ...)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the random blocks of the non-uniform families")
    parser.add_argument("--verify", type=int, default=25,
                        help="Verify function preservation on this many INRs per width "
                             "(0 disables)")
    args = parser.parse_args()

    suffix = args.out_suffix or FAMILY_SUFFIX[args.family]
    subdir = "fmnist_inrs" if args.dataset == "fmnist" else "mnist_inrs"
    split_name = args.split_name or f"{args.dataset}_sizegen_splits.json"
    data_root = Path(os.environ.get("ANYDIM_DATA_ROOT", "data")) / subdir
    base_dir = data_root / f"w{args.base_width}_{args.init_type}"
    split_path = base_dir / split_name

    with open(split_path) as f:
        splits = json.load(f)

    test_paths = splits["test"]["path"]
    test_labels = splits["test"]["label"]
    print(f"Base: w{args.base_width}_{args.init_type}, {len(test_paths)} test INRs")
    print(f"Family: {args.family}  ->  w{{N}}_{args.init_type}_{suffix}/  (seed {args.seed})")

    coords = make_coord_grid(28, 28) if args.verify else None

    for target_width in args.target_widths:
        assert target_width % args.base_width == 0, \
            f"Target width {target_width} must be a multiple of base width {args.base_width}"
        k = target_width // args.base_width

        # Clear the target dir first: the same w{N}_{init}_{suffix}/ name is reused across
        # base widths, and only the files named by the current split get overwritten.
        # Without this, a rerun at a different base width leaves the previous run's INRs
        # behind and the directory holds two datasets at once.
        out_dir = data_root / f"w{target_width}_{args.init_type}_{suffix}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        # One generator per width, so a width can be regenerated on its own reproducibly.
        gen = torch.Generator().manual_seed(args.seed + target_width)

        out_paths = []
        logit_err = pixel_err = row_err = col_err = 0.0
        nonuniformity = float("inf")
        for i, src_path in enumerate(test_paths):
            src_path = Path(src_path)

            # Mirror directory structure: mnist_png_train/{label}/{filename}
            rel = src_path.relative_to(base_dir)
            dst_path = out_dir / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            sd = torch.load(src_path, map_location="cpu", weights_only=True)
            widened_sd = widen_state_dict(sd, k, args.family, gen)
            torch.save(widened_sd, dst_path)
            out_paths.append(str(dst_path))

            if i < args.verify:
                z_base, y_base = forward_inr(sd, coords)
                z_wide, y_wide = forward_inr(widened_sd, coords)
                logit_err = max(logit_err, (z_base - z_wide).abs().max().item())
                pixel_err = max(pixel_err, (y_base - y_wide).abs().max().item())
                r, c, nu = block_condition_errors(sd, widened_sd, k)
                row_err, col_err = max(row_err, r), max(col_err, c)
                nonuniformity = min(nonuniformity, nu)

        # Write split JSON (test-only, same labels)
        dup_splits = {
            "train": {"path": [], "label": []},
            "val": {"path": [], "label": []},
            "test": {"path": out_paths, "label": test_labels},
        }
        with open(out_dir / split_name, "w") as f:
            json.dump(dup_splits, f)

        print(f"  w{target_width} (k={k}): wrote {len(out_paths)} files to {out_dir}")
        if args.verify:
            print(f"    verify on {min(args.verify, len(test_paths))} INRs: "
                  f"max|Delta pixels| = {pixel_err:.2e}, max|Delta logits| = {logit_err:.2e}"
                  f"  (row {row_err:.2e}, col {col_err:.2e}, "
                  f"min nonuniformity {nonuniformity:.2e})")

    print("Done.")


if __name__ == "__main__":
    main()
