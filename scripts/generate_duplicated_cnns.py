"""
Generate widened copies of the small CNN zoo's ReLU test models, in any of the five
channel-widening equivalence families.

Stage 2 of the CNN accuracy-prediction study — the analog of
``scripts/generate_duplicated_inrs.py`` / Experiment 2 on the INR side.  **No CNN is
trained here**: each widened network represents *exactly the same function* as its
w16 original (verified by ``scripts/check_cnn_widening_equiv.py``), so its test
accuracy is the original's, copied unchanged.

That makes it a clean, assumption-free test of one thing only: does the
metanetwork's prediction stay put when the *same function* is presented at a wider
architecture?  A model invariant to that widening family must give the identical
prediction, hence Kendall tau = 1.000 and identical R2 / L1 at every width; a model
that is not invariant need not.  There is no SP/muP distinction, because no wide
model is ever trained.

Widening (``src.data.zoo_cnn.widen_wb``): with ``W^(l)`` in ``R^{n_out x n_in x kh x kw}``
the widened kernel is written blockwise, per kernel offset ``s``,

    W_up[i*k_out + a, j*k_in + b, s] = B_{ij}[a, b, s],   b_up = b (x) 1_{k_out},

and the function is preserved iff every block's row sums equal the base kernel,

    (row)  sum_b B_{ij}[a, b, s] = W[i, j, s]      for every a, s

which is the only condition the forward matrix-product aggregate ever sees.  The
backward aggregate uses ``W^T`` and additionally needs the column condition
``sum_a B_{ij}[a, b, s] = (k_out / k_in) W[i, j, s]``.  ``--family`` picks which
subspace the blocks are drawn from (``FAMILIES``, defined once in
``src/data/zoo_cnn.py``); ``uniform`` is the Kronecker special case
``B_{ij} = W_ij 1 1^T / k_in`` (row **and** column) and ``general`` is the largest
function-preserving family (row only).

Blocks are built in **float64** and cast to float32 on write: the free entries are O(1)
randn draws whose row sums cancel down to the O(1) base kernel, and in float32 that
cancellation leaves ~1e-7 of absolute error on a ~1e-2 weight.

Output layout, one directory per width (``svhn_cnn_zoo/`` for ``--dataset svhn``)::

    $ANYDIM_DATA_ROOT/cnn_zoo/w{N}_zoo_dup/          (uniform; `_zoo_gen` for general, ...)
        models/{split}/{idx:06d}.pth       layers.{i}.{weight,bias} state dicts
        cnn_zoo_dup_splits.json            {"train": {...}, "val": {...}, "test": {...}}

``k = 1`` is family-independent (the constraints pin every block to the base kernel), so
it is written only for ``--family uniform``: every family reads its base-width column out
of ``w16_zoo_dup/``.  That copy gets **all three** splits rather than just ``test``.  Two
reasons:

* the train/val splits are what the Stage 2 metanetworks are trained on, so the
  whole of Stage 2 goes through one reader
  (:class:`src.data.cnn_zoo_dataset.CNNZooDataset`) instead of mixing it with
  upstream's ``NFNZooDataset``;
* the base-width test number then differs from Stage 1's only by the export, which
  makes it a round-trip check on the ``.pth`` export itself.

The split JSON is *merged*, not overwritten, so invocations with different
``--splits`` accumulate into one file.

The split JSON also carries ``score_ce`` (test cross-entropy) and ``score_train_acc``
alongside ``score`` (test accuracy).  ``CNNZooDataset`` reads only ``path``/``score``;
the extra keys are there so the theory-aligned secondary target is available without
regenerating anything.

Usage
-----
    python scripts/generate_duplicated_cnns.py                       # k = 1,2,3,4,6,8
    python scripts/generate_duplicated_cnns.py --k 1 2 4 --max-models 500
    python scripts/generate_duplicated_cnns.py --max-models -1       # all test models
    python scripts/generate_duplicated_cnns.py --dataset svhn        # SVHN-GS zoo
    python scripts/generate_duplicated_cnns.py --family general      # -> w{N}_zoo_gen/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCALEGMN_ROOT = PROJECT_ROOT / "src" / "scalegmn"

sys.path.insert(0, str(SCALEGMN_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.zoo_cnn import (  # noqa: E402
    FAMILIES, check_family_conditions, layer_layout, uniform_multipliers,
    wb_to_state_dict, widen_wb, zoo_cnn_forward,
)

BASE_WIDTH = 16

# Directory tag per family. `uniform` keeps `zoo_dup` for backwards compatibility with the
# w{16..128}_zoo_dup/ datasets already on disk (both zoos).
FAMILY_SUFFIX = {
    "uniform": "zoo_dup",
    "row-stoch": "zoo_rowst",
    "doubly-stoch": "zoo_dblst",
    "general-bidir": "zoo_genbd",
    "general": "zoo_gen",
}

# Blocks are drawn in double; see the module docstring for why float32 is not enough.
DTYPE = torch.float64


def split_name(suffix: str) -> str:
    """Split JSON filename for a family's tree (`cnn_zoo_dup_splits.json` for uniform)."""
    return f"cnn_{suffix}_splits.json"


# Kept for importers that predate --family; the uniform tree's name is unchanged.
SPLIT_NAME = split_name(FAMILY_SUFFIX["uniform"])


def data_root() -> Path:
    root = os.environ.get("ANYDIM_DATA_ROOT")
    if root is None:
        raise EnvironmentError("ANYDIM_DATA_ROOT is not set — run `source .env` first.")
    return Path(root)


def load_zoo_split(zoo_dir: Path, split: str, activation: str, max_models: int,
                   dataset: str = "cifar10"):
    """Per-model weight/bias lists + metrics for one split of the zoo.

    Delegates every filtering decision (final-step selection, the NFN shuffle order,
    the iid split, the activation filter, the TF->PyTorch kernel transpose) to
    upstream's :class:`NFNZooDataset`, so the models here are bit-identical to the
    ones Stage 1 trains and evaluates on.
    """
    os.chdir(SCALEGMN_ROOT)  # upstream imports assume this CWD
    from src.data.cifar10_dataset import NFNZooDataset  # noqa: E402

    ds = NFNZooDataset(
        dataset=dataset,
        dataset_path=str(zoo_dir),
        data_path=str(zoo_dir / "weights.npy"),
        metrics_path=str(zoo_dir / "metrics.csv.gz"),
        layout_path=str(zoo_dir / "layout.csv"),
        idcs_file=str(zoo_dir / f"{dataset}_split.csv"),
        split=split,
        activation_function=activation,
        layer_layout=layer_layout(BASE_WIDTH),
        data_format="nfn",
    )
    n = len(ds)
    if max_models is not None and max_models > 0:
        n = min(n, max_models)
    print(f"zoo split={split} activation={activation}: {len(ds)} models, using {n}")
    return ds, n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["cifar10", "svhn"], default="cifar10",
                   help="which upstream zoo to widen; also picks the default in/out trees")
    p.add_argument("--zoo-dir", default=None,
                   help="default $ANYDIM_DATA_ROOT/{dataset}_zoo")
    p.add_argument("--out-root", default=None,
                   help="default $ANYDIM_DATA_ROOT/cnn_zoo (svhn: svhn_cnn_zoo)")
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8],
                   help="width multipliers (k=1 writes the un-widened base set)")
    p.add_argument("--splits", nargs="+", default=["test"],
                   choices=["train", "val", "test"],
                   help="splits to widen at k > 1")
    p.add_argument("--base-splits", nargs="+", default=["train", "val", "test"],
                   choices=["train", "val", "test"],
                   help="splits to export at k = 1 (the metanetwork's training data)")
    p.add_argument("--family", choices=FAMILIES, default="uniform",
                   help="widening equivalence class (default: uniform Kronecker); picks "
                        "the output tree, w{N}_zoo_dup/ -> w{N}_zoo_gen/ etc.")
    p.add_argument("--out-suffix", default=None,
                   help="override the per-family output dir tag (default: uniform->"
                        "zoo_dup, general->zoo_gen, ...)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the random blocks of the non-uniform families")
    p.add_argument("--activation", default="relu")
    p.add_argument("--max-models", type=int, default=2000,
                   help="cap the number of widened models per split (-1 = all). The "
                        "default keeps the whole Stage 2 set at a few GB; tau on 2000 "
                        "models is far more precise than the effect being measured. "
                        "Ignored for the k=1 train/val splits, which are always full.")
    p.add_argument("--check-every", type=int, default=200,
                   help="verify functional equivalence on every Nth model (0 = never)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    root = data_root()
    # svhn writes to its own tree so the two studies' w{N}_zoo_dup/ dirs cannot collide
    out_tree = "cnn_zoo" if args.dataset == "cifar10" else f"{args.dataset}_cnn_zoo"
    zoo_dir = Path(args.zoo_dir) if args.zoo_dir else root / f"{args.dataset}_zoo"
    out_root = Path(args.out_root) if args.out_root else root / out_tree
    suffix = args.out_suffix or FAMILY_SUFFIX[args.family]
    SPLIT = split_name(suffix)
    print(f"dataset={args.dataset}  zoo={zoo_dir}  out={out_root}")
    print(f"family={args.family}  ->  w{{N}}_{suffix}/  (seed {args.seed}, {SPLIT})")

    # k = 1 is family-independent: (row) with k_in = 1 pins every block to the base kernel.
    # Only the uniform tree carries it, so every family shares one base-width column.
    if args.family != "uniform" and 1 in args.k:
        args.k = [k for k in args.k if k != 1]
        print(f"[note] dropped k=1: it is family-independent, so the base-width copy lives "
              f"in w{BASE_WIDTH}_{FAMILY_SUFFIX['uniform']}/ for every family. k={args.k}")

    x_probe = torch.randn(2, 1, 32, 32, dtype=DTYPE)
    needed = sorted({s for k in args.k
                     for s in (args.base_splits if k == 1 else args.splits)})
    # one NFNZooDataset per split, reused across widths (loading weights.npy is the
    # expensive part)
    zoo = {}
    for split in needed:
        # the metanetwork trains on all of train/val; only the evaluated split is capped
        cap = -1 if split in ("train", "val") else args.max_models
        zoo[split] = load_zoo_split(zoo_dir, split, args.activation, cap, args.dataset)

    for k in args.k:
        width = BASE_WIDTH * k
        out_dir = out_root / f"w{width}_{suffix}"
        split_json = out_dir / SPLIT
        expected_layout = layer_layout(width)
        # one generator per width, so a width can be regenerated on its own reproducibly
        gen = torch.Generator().manual_seed(args.seed + width)

        records = {}
        if split_json.exists():
            with open(split_json) as f:
                records = json.load(f)

        for split in (args.base_splits if k == 1 else args.splits):
            if split in records and not args.overwrite:
                print(f"[skip] w{width} {split}: already in {split_json.name} "
                      "(use --overwrite)")
                continue
            ds, n_models = zoo[split]
            models_dir = out_dir / "models" / split
            models_dir.mkdir(parents=True, exist_ok=True)

            paths, scores, ces, train_accs = [], [], [], []
            max_fn_err = row_err = col_err = 0.0
            nonuniformity = float("inf")

            for i in tqdm(range(n_models), desc=f"w{width} (k={k}) {split}"):
                item = ds[i]
                weights = [w.double() for w in item.weights]
                biases = [b.double() for b in item.biases]
                row = ds.metrics.iloc[i]

                mults = uniform_multipliers(len(weights), k)
                # float64 in, float32 on write: the general families' blocks are O(1)
                # randn draws whose row sums cancel to the O(1e-2) base kernel
                wide_w, wide_b = widen_wb(weights, biases, mults, family=args.family,
                                          generator=gen)

                got = [wide_w[0].shape[1]] + [b.shape[0] for b in wide_b]
                assert got == expected_layout, f"layout {got} != {expected_layout}"

                wide_w = [w.float() for w in wide_w]
                wide_b = [b.float() for b in wide_b]

                # everything is checked on the float32 tensors that actually get stored,
                # not on the float64 construction
                if args.check_every and i % args.check_every == 0:
                    f_base = zoo_cnn_forward(weights, biases, x_probe)
                    f_wide = zoo_cnn_forward([w.double() for w in wide_w],
                                             [b.double() for b in wide_b], x_probe)
                    max_fn_err = max(max_fn_err, (f_base - f_wide).abs().max().item())
                    # residuals of the two block conditions, plus the distance from the
                    # uniform block: the MINIMUM over models, so a family that silently
                    # degenerated to Kronecker cannot pass unnoticed
                    r, c, nu = check_family_conditions(
                        [w.float() for w in weights], wide_w, mults)
                    row_err, col_err = max(row_err, r), max(col_err, c)
                    nonuniformity = min(nonuniformity, nu)

                rel = f"models/{split}/{i:06d}.pth"
                torch.save(wb_to_state_dict(wide_w, wide_b), out_dir / rel)
                paths.append(rel)
                scores.append(float(row.test_accuracy))
                ces.append(float(row.test_loss))
                train_accs.append(float(row.train_accuracy))

            records[split] = {"path": paths, "score": scores,
                              "score_ce": ces, "score_train_acc": train_accs}
            with open(split_json, "w") as f:
                json.dump(records, f)

            size_mb = sum(f.stat().st_size for f in models_dir.glob("*.pth")) / 2 ** 20
            # uniform, k=3, 6: the 1/k_in factor is not exactly representable in float32,
            # so the stored widened weights carry a ~1e-7 relative rounding error
            print(f"w{width} {split}: {len(paths)} models, {size_mb:.0f} MB, "
                  f"layout={expected_layout}, max |f_base - f_wide| = {max_fn_err:.2e}")
            if args.check_every:
                print(f"    conditions on the stored float32 weights: row {row_err:.2e}, "
                      f"col {col_err:.2e}, min nonuniformity {nonuniformity:.2e}")
        print(f"  -> {split_json}")

    print("\nDone. Evaluate with scripts/eval_sizegen_predgen_duplicated.py")


if __name__ == "__main__":
    main()
