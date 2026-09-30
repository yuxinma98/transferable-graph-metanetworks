"""
Input-network quality of the multi-width CNN zoo — the CNN analogue of
``verify_inr_quality.py`` + ``verify_norm_stats.py``.

Everything here is read off the ``meta_{role}.jsonl`` records written by
``generate_cnn_zoo.py``; no model is loaded and no CNN is re-run.  Those records
already carry the label (``test_acc``), the sampled hyperparameters (``hp``) and the
per-layer **normalized** spectral norms ``sqrt(n_{l-1}/n_l) * ||W^(l)||_2`` of the
*effective* weights (convs flattened to ``[c_out, c_in*kh*kw]``).

Four tables, matching diagnostics 1, 5, 2 and 7 of
``results/EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md``:

  1. per-width label distributions          -- is a tau drop a support problem?
  2. per-width normalized spectral norms    -- is muP16 actually the flat arm?
  3. tau_HP(w0, w) on the paired set        -- how much of the ranking transfers at all?
  4. per-width tau_b ceiling                -- is a tau drop just the metric's own ceiling?

``||w||_{V_n} = sum_l sqrt(n_{l-1}/n_l) * ||W^(l)||_2`` is the per-model scalar of
table 2, the same convention as the INR-side norm statistics, and ``W1(mu_w0, mu_w)`` is
the 1-Wasserstein distance between its w0 and w distributions.  The paired role is drawn
from a width-independent seed, so index ``i`` is the *same* hyperparameter draw at every
width, which is what makes table 3 a paired comparison.

Usage
-----
    python scripts/summarize_cnn_zoo_quality.py
    python scripts/summarize_cnn_zoo_quality.py --role paired --arms sp
    python scripts/summarize_cnn_zoo_quality.py --dataset svhn   # the SVHN-GS replication
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, wasserstein_distance

#: Test accuracy a CNN that learned nothing lands on.  Uniform chance for balanced
#: CIFAR-10-GS, but SVHN's test split is imbalanced, so its failed draws pile up on the
#: majority class (digit "1", 5099/26032) rather than at 0.1 — reading the SVHN zoo against
#: 0.1 counts 0.02% of models as failed where 47% actually are.
FLOORS = {"cifar10": 0.1, "svhn": 5099 / 26032}

#: Dataset-dependent globals, rebound once in :func:`main` from ``--dataset`` rather than
#: threaded through all five table functions, which only ever read one study's records per
#: invocation.  ``ZOO_TREE`` is the tree under ``$ANYDIM_DATA_ROOT``.
ZOO_TREE = "cnn_zoo"
FLOOR = FLOORS["cifar10"]


def set_dataset(dataset: str, floor: float | None = None) -> None:
    """Rebind the dataset-dependent globals — the only supported way to switch study.

    Importers (``render_predgen_v2_results.py``) call this instead of assigning the
    globals themselves, so the tree/floor pairing can never come apart.
    """
    global ZOO_TREE, FLOOR
    ZOO_TREE = "cnn_zoo" if dataset == "cifar10" else f"{dataset}_cnn_zoo"
    FLOOR = FLOORS[dataset] if floor is None else floor


def data_root() -> Path:
    root = os.environ.get("ANYDIM_DATA_ROOT")
    if root is None:
        raise EnvironmentError("ANYDIM_DATA_ROOT is not set — run `source .env` first.")
    return Path(root)


def read_meta(width: int, arm: str, role: str, tree: str | None = None) -> list[dict]:
    """One role's records, ordered by ``idx`` (the split's order)."""
    path = data_root() / (tree or ZOO_TREE) / f"w{width}_{arm}" / f"meta_{role}.jsonl"
    if not path.exists():
        return []
    recs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["idx"]] = r
    return [recs[i] for i in sorted(recs)]


def labels(recs) -> np.ndarray:
    return np.array([r["test_acc"] for r in recs])


def norms(recs) -> np.ndarray:
    """``||w||_{V_n}`` per model: the sum of the per-layer normalized spectral norms."""
    return np.array([sum(r["spec_norms"]) for r in recs])


def layer_norms(recs) -> np.ndarray:
    """``[n_models, n_layers]`` of per-layer normalized spectral norms."""
    return np.array([r["spec_norms"] for r in recs])


def label_table(widths, arms, role):
    print(f"\n### Per-width label distributions (`{role}`, test accuracy)\n")
    print(f"| Width | arm | n | min | q25 | median | q75 | max | mean | "
          f"frac ≤ {FLOOR + 0.01:.3f} |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        for w in widths:
            y = labels(read_meta(w, arm, role))
            if not len(y):
                print(f"| w{w} | {arm} | — | | | | | | | |")
                continue
            q = np.percentile(y, [0, 25, 50, 75, 100])
            print(f"| w{w} | {arm} | {len(y)} | {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} | "
                  f"{q[3]:.3f} | {q[4]:.3f} | {y.mean():.3f} | "
                  f"{(y <= FLOOR + 0.01).mean():.3f} |")


def overlap_table(widths, arms, role, base):
    """How much of each width's label distribution sits inside the base width's range."""
    print(f"\n### Label-support overlap with w{base} (`{role}`)\n")
    print("| Width | arm | W₁(y_base, y_w) | frac inside base [q1, q99] | median shift |")
    print("|---|---|---|---|---|")
    for arm in arms:
        y0 = labels(read_meta(base, arm, role))
        lo, hi = np.percentile(y0, [1, 99])
        for w in widths:
            y = labels(read_meta(w, arm, role))
            if not len(y) or not len(y0):
                continue
            inside = ((y >= lo) & (y <= hi)).mean()
            print(f"| w{w} | {arm} | {wasserstein_distance(y0, y):.4f} | {inside:.3f} | "
                  f"{np.median(y) - np.median(y0):+.4f} |")


def tau_b_ceiling(width: int, arm: str, role: str = "test") -> tuple[float, float]:
    """``(tied-pair fraction, largest tau_b a perfect ranker can score)`` on one cohort.

    A pair tied in ``y`` is neither concordant nor discordant, so it leaves the numerator of
    ``tau_b = (C - D) / sqrt((n_0 - n_1)(n_0 - n_2))`` untouched while being discounted in
    only *one* factor of the denominator.  With continuous predictions (``n_1 = 0``) and a
    perfect ranking (``C = n_0 - n_2``, ``D = 0``) what survives is ``sqrt((n_0-n_2)/n_0)``,
    which is below 1 whenever labels tie -- and on SVHN they tie heavily, since every CNN
    that fails to learn scores *exactly* the majority-class rate and so joins one big tie
    block.  Ties are counted by exact float equality, which is what that block is.
    """
    return _ceiling(ZOO_TREE, width, arm, role)


@lru_cache(maxsize=None)
def _ceiling(tree, width, arm, role):
    y = labels(read_meta(width, arm, role, tree))
    if len(y) < 2:
        return float("nan"), float("nan")
    counts = np.unique(y, return_counts=True)[1]
    n_0 = len(y) * (len(y) - 1) / 2
    n_2 = (counts * (counts - 1) / 2).sum()
    return float(n_2 / n_0), float(np.sqrt((n_0 - n_2) / n_0))


def ceiling_table(widths, arms, role):
    """Which per-width metric comparisons are safe, and against what reference.

    ``tau_b_max`` is the ceiling above; ``sd(y)`` is R^2's *denominator*, so it is the other
    half of the same question.  R^2 has no tie ceiling (a perfect predictor scores 1 at every
    width) but it is measured against a per-width null -- predicting that width's own mean --
    so a width whose labels spread out flatters equal absolute error.  L1 is normalized by
    nothing and is the one raw number that compares across widths as-is.
    """
    print(f"\n### Per-width $\\tau_b$ ceiling and R^2 reference (`{role}`)\n")
    print("| Width | " + " | ".join(
        f"{arm} tie frac | {arm} τ_b max | {arm} sd(y)" for arm in arms) + " |")
    print("|---" * (1 + 3 * len(arms)) + "|")
    for w in widths:
        cells = []
        for arm in arms:
            tie, ceil = tau_b_ceiling(w, arm, role)
            y = labels(read_meta(w, arm, role))
            cells += ["", "", ""] if np.isnan(ceil) else [
                f"{tie:.4f}", f"{ceil:.4f}", f"{y.std(ddof=1):.4f}"]
        print(f"| w{w} | " + " | ".join(cells) + " |")


def norm_table(widths, arms, role, base):
    print(f"\n### Per-width normalized spectral norms (`{role}`, "
          f"‖w‖_Vn = Σ_l √(n_{{l−1}}/n_l)·‖W^(l)‖₂)\n")
    cols = " | ".join(f"{arm} mean ‖w‖_Vn | {arm} W₁(μ_base, μ_w)" for arm in arms)
    print(f"| Width | {cols} |")
    print("|---" * (1 + 2 * len(arms)) + "|")
    base_norms = {arm: norms(read_meta(base, arm, role)) for arm in arms}
    for w in widths:
        cells = []
        for arm in arms:
            n = norms(read_meta(w, arm, role))
            b = base_norms[arm]
            if not len(n) or not len(b):
                cells += ["", ""]
            elif w == base:
                cells += [f"{n.mean():.2f}", "0 (by definition)"]
            else:
                cells += [f"{n.mean():.2f}", f"{wasserstein_distance(b, n):.2f}"]
        print(f"| w{w} | " + " | ".join(cells) + " |")

    print(f"\nPer-layer means (conv1 / conv2 / conv3 / fc), `{role}`:\n")
    for arm in arms:
        for w in widths:
            ln = layer_norms(read_meta(w, arm, role))
            if not len(ln):
                continue
            print(f"- {arm} w{w}: " + " / ".join(f"{v:.2f}" for v in ln.mean(0)))


def tau_hp_table(widths, arms, base):
    """Diagnostic 2: how much of the accuracy ranking survives a width change at all."""
    print(f"\n### τ_HP(w{base}, w) — same hyperparameter draw, two widths (`paired`)\n")
    print("| Width | " + " | ".join(f"{arm} τ_HP" for arm in arms) + " |")
    print("|---" * (1 + len(arms)) + "|")
    for w in widths:
        cells = []
        for arm in arms:
            y0 = labels(read_meta(base, arm, "paired"))
            y = labels(read_meta(w, arm, "paired"))
            if not len(y) or not len(y0) or len(y) != len(y0):
                cells.append("")
            else:
                cells.append(f"{kendalltau(y0, y).statistic:.4f}")
        print(f"| w{w} | " + " | ".join(cells) + " |")


def lr_strata(widths, arms, role, nbins=4):
    """Mean label per base-LR quartile: the accuracy-vs-LR curve, compactly."""
    print(f"\n### Mean label by sampled-LR quartile (`{role}`)\n")
    for arm in arms:
        base_recs = read_meta(widths[0], arm, role)
        if not base_recs:
            continue
        lrs = np.array([r["hp"]["lr"] for r in base_recs])
        edges = np.percentile(lrs, np.linspace(0, 100, nbins + 1))
        print(f"- {arm} LR quartile edges: " + " / ".join(f"{e:.1e}" for e in edges))
        for w in widths:
            recs = read_meta(w, arm, role)
            if not recs:
                continue
            y = labels(recs)
            lr = np.array([r["hp"]["lr"] for r in recs])
            b = np.clip(np.digitize(lr, edges[1:-1]), 0, nbins - 1)
            means = [y[b == k].mean() if (b == k).any() else float("nan")
                     for k in range(nbins)]
            print(f"  - w{w}: " + " / ".join(f"{m:.3f}" for m in means))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--widths", type=int, nargs="+",
                   default=[16, 32, 48, 64, 96, 128, 192, 256, 384, 512])
    p.add_argument("--arms", nargs="+", default=["sp", "mup16"])
    p.add_argument("--role", default="test", help="role the per-width tables read")
    p.add_argument("--base-width", type=int, default=16)
    p.add_argument("--dataset", choices=sorted(FLOORS), default="cifar10",
                   help="which study's zoo to summarize (cifar10 -> cnn_zoo/, "
                        "svhn -> svhn_cnn_zoo/)")
    p.add_argument("--floor", type=float, default=None,
                   help="override the no-learning accuracy floor (default: per --dataset)")
    args = p.parse_args()

    set_dataset(args.dataset, args.floor)

    print(f"CNN zoo quality — {args.dataset}, arms {args.arms}, widths {args.widths}, "
          f"role '{args.role}', base w{args.base_width}, floor {FLOOR:.4f}")
    label_table(args.widths, args.arms, args.role)
    overlap_table(args.widths, args.arms, args.role, args.base_width)
    ceiling_table(args.widths, args.arms, args.role)
    norm_table(args.widths, args.arms, args.role, args.base_width)
    tau_hp_table(args.widths, args.arms, args.base_width)
    lr_strata(args.widths, args.arms, args.role)


if __name__ == "__main__":
    main()
