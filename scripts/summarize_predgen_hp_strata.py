#!/usr/bin/env python
"""Mandatory diagnostic 4: Kendall tau computed *within* hyperparameter strata.

The per-width tau in the results docs is measured across a test cohort whose CNNs were
drawn with different hyperparameters, so a predictor can score well by reading the
hyperparameters off the weights (learning rate leaves a scale signature, epochs a
sharpness one) without ranking two same-recipe networks correctly. Restricting the
comparison to pairs inside one stratum removes that shortcut: what survives is only what
the weights carry beyond the recipe. Read against `tau_HP(16, w)` (the ceiling a
hyperparameter-only predictor reaches, `summarize_cnn_zoo_quality.py`), this closes the
question that table opens -- tau_HP says how much signal the recipe alone gives, and the
stratified tau here says how much a model has on top of it.

The prediction dumps carry only `w{N}_{pred,actual}`; the hyperparameters live in each
width's split JSON, which is also what fixes the dump's row order (`CNNZooDataset` reads
`[split]["path"]` in file order and the test loaders are `shuffle=False`), so the join is
positional. `w{N}_actual` is verified against that split's `score` before anything is
reported.

Two stratifications, both computed on every width of every condition:

  lr   10 log-spaced learning-rate deciles (~100 CNNs each). The learning rate is the
       one continuously-sampled hyperparameter that drives accuracy hardest, so this is
       the "fixed-LR strata" reading.
  hp   the discrete recipe (dropout, epochs, train_frac) matched exactly and crossed
       with learning-rate terciles -- a stricter stratum (~9 CNNs each) at the cost of
       far fewer comparable pairs.

Within a stratification, tau is pooled the way tau_b itself is defined rather than
averaged over strata: concordant minus discordant summed over within-stratum pairs, over
the summed per-stratum tau_b denominators. That keeps unequal strata weighted by how many
comparable pairs they actually contribute and handles tied accuracies (which SVHN has a
lot of) identically to the unstratified number.

Usage:
    python scripts/summarize_predgen_hp_strata.py --dataset cifar10
    python scripts/summarize_predgen_hp_strata.py --dataset svhn --strata lr
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

# (label, directory, filename-glob) per arm of the study. Populated per dataset below.
ZOO_TREE = {"cifar10": "cnn_zoo", "svhn": "svhn_cnn_zoo"}
SPLIT_JSON = {"cifar10": "cifar10_predgen_splits.json", "svhn": "svhn_predgen_splits.json"}
PRED_DIRS = {
    "cifar10": [
        ("forward", "/tmp/predgen_sizegen_v1"),
        ("forward", "/tmp/predgen_sizegen_v2"),
        ("bidir", "/tmp/predgen_sizegen_v1_bidir"),
        ("bidir", "/tmp/predgen_sizegen_v2_bidir"),
        ("mpsgmn", "/tmp/mpsgmn_predgen"),
    ],
    "svhn": [
        ("forward", "/tmp/predgen_svhn_sizegen_v1"),
        ("forward", "/tmp/predgen_svhn_sizegen_v2"),
        ("bidir", "/tmp/predgen_svhn_sizegen_v1_bidir"),
        ("bidir", "/tmp/predgen_svhn_sizegen_v2_bidir"),
        ("mpsgmn", "/tmp/mpsgmn_predgen_svhn"),
    ],
}

# Bidirectional `symmetry: scale` was trained before the reciprocal backward edge feature
# reached the predgen entry points, so those checkpoints are not scale-equivariant and are
# out of scope on both zoos (see the results docs' stage 3f, and `plot_sizegen_results.py`,
# which likewise never generates their cells). Their dumps exist; they are never reported.
OUT_OF_SCOPE = {"sgmn-base-bidir", "sgmn-deq-bidir"}


def arm_of(stem: str) -> str:
    """Which parameterization arm's zoo a dump was scored against."""
    if "mup16" in stem:
        return "mup16"
    if "sp" in stem:
        return "sp"
    raise ValueError(f"cannot tell the arm from {stem!r}")


def condition_of(stem: str) -> str:
    """Human label for a dump, arm stripped out (the arm is a separate column)."""
    tokens = re.split(r"[-_]", stem[len("preds_"):])
    return "-".join(t for t in tokens if t not in ("mup16", "sp"))


def load_hp(root: Path, dataset: str, width: int, arm: str) -> tuple[list[dict], np.ndarray]:
    """The test role's hyperparameters and scores for one (width, arm), in dump order."""
    path = root / ZOO_TREE[dataset] / f"w{width}_{arm}" / SPLIT_JSON[dataset]
    with open(path) as fh:
        test = json.load(fh)["test"]
    return test["hp"], np.asarray(test["score"], dtype=float)


def strata_lr(hp: list[dict], n_bins: int = 10) -> np.ndarray:
    """Learning-rate quantile bins, equal-count by construction."""
    lr = np.log10([h["lr"] for h in hp])
    edges = np.quantile(lr, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, lr)


def strata_hp(hp: list[dict], n_lr_bins: int = 3) -> np.ndarray:
    """Exact (dropout, epochs, train_frac) cell crossed with learning-rate terciles."""
    lr_bin = strata_lr(hp, n_lr_bins)
    keys = [(h["dropout"], h["epochs"], h["train_frac"], b) for h, b in zip(hp, lr_bin)]
    order = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.array([order[k] for k in keys])


def tau_b_parts(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """``(C - D, tau_b denominator)`` for one group, via explicit pair enumeration.

    Groups here are <= a few hundred items, so the O(m^2) sign matrix is cheaper than
    any sort-based scheme and makes the tie bookkeeping obvious.
    """
    m = len(actual)
    if m < 2:
        return 0.0, 0.0
    upper = np.triu(np.ones((m, m), dtype=bool), k=1)
    da = np.sign(actual[:, None] - actual[None, :])[upper]
    dp = np.sign(pred[:, None] - pred[None, :])[upper]
    agree = da * dp
    n0 = float(upper.sum())
    n1 = float((da == 0).sum())   # pairs tied in the true accuracy
    n2 = float((dp == 0).sum())   # pairs tied in the prediction
    denom = np.sqrt(max(n0 - n1, 0.0) * max(n0 - n2, 0.0))
    return float((agree > 0).sum() - (agree < 0).sum()), float(denom)


def stratified_tau(actual: np.ndarray, pred: np.ndarray,
                   labels: np.ndarray) -> tuple[float, int, float]:
    """Pooled within-stratum tau_b, plus the stratum count and mean stratum size."""
    num = den = 0.0
    sizes = []
    for s in np.unique(labels):
        idx = labels == s
        sizes.append(int(idx.sum()))
        cd, d = tau_b_parts(actual[idx], pred[idx])
        num += cd
        den += d
    tau = num / den if den > 0 else float("nan")
    return tau, len(sizes), float(np.mean(sizes)) if sizes else float("nan")


def analyse(npz_path: Path, dataset: str, root: Path, strata: list[str]) -> dict | None:
    stem = npz_path.stem
    arm = arm_of(stem)
    data = np.load(npz_path)
    widths = sorted({int(k[1:].split("_")[0]) for k in data.files})
    rows = []
    for w in widths:
        pred = data[f"w{w}_pred"].astype(float).ravel()
        actual = data[f"w{w}_actual"].astype(float).ravel()
        hp, score = load_hp(root, dataset, w, arm)
        if len(hp) != len(actual):
            print(f"  SKIP w{w}: split has {len(hp)} rows, dump has {len(actual)}",
                  file=sys.stderr)
            continue
        if not np.allclose(actual, score, atol=1e-5):
            raise SystemExit(
                f"{npz_path}: w{w} actuals do not match {arm} split scores — the "
                f"positional join is not valid here (max |d| = "
                f"{np.abs(actual - score).max():.3g})")
        row = {"width": w, "tau": float(kendalltau(actual, pred).correlation)}
        for kind in strata:
            labels = strata_lr(hp) if kind == "lr" else strata_hp(hp)
            tau_s, n_strata, mean_size = stratified_tau(actual, pred, labels)
            row[f"tau_{kind}"] = tau_s
            row[f"n_{kind}"] = n_strata
            row[f"size_{kind}"] = mean_size
        rows.append(row)
    if not rows:
        return None
    return {"condition": condition_of(stem), "arm": arm, "rows": rows}


def render(results: list[dict], strata: list[str], title: str) -> str:
    out = [f"# {title}", ""]
    for res in results:
        out += [f"### {res['condition']} — {res['arm']}", ""]
        head = "| Width | tau |" + "".join(f" tau ({k}-strat) | delta |" for k in strata)
        rule = "|---|---|" + "---|---|" * len(strata)
        out += [head, rule]
        for r in res["rows"]:
            cells = [f"w{r['width']}", f"{r['tau']:+.4f}"]
            for k in strata:
                cells += [f"{r[f'tau_{k}']:+.4f}", f"{r[f'tau_{k}'] - r['tau']:+.4f}"]
            out.append("| " + " | ".join(cells) + " |")
        ood = [r for r in res["rows"] if r["width"] != res["rows"][0]["width"]]
        if ood:
            cells = ["**OOD mean**", f"**{np.mean([r['tau'] for r in ood]):+.4f}**"]
            for k in strata:
                ms = np.mean([r[f"tau_{k}"] for r in ood])
                cells += [f"**{ms:+.4f}**",
                          f"**{ms - np.mean([r['tau'] for r in ood]):+.4f}**"]
            out.append("| " + " | ".join(cells) + " |")
        sizes = ", ".join(f"{k}: {res['rows'][0][f'n_{k}']} strata, "
                          f"mean {res['rows'][0][f'size_{k}']:.0f} CNNs" for k in strata)
        out += ["", f"Strata — {sizes}.", ""]
    return "\n".join(out)


def render_compact(results: list[dict], strata: list[str], title: str) -> str:
    """One row per (condition, arm): in-distribution tau and the OOD means, for the docs."""
    out = [f"# {title}", ""]
    head = ("| Condition | Arm | $\\tau$ (w{first}) | OOD $\\tau$ |"
            + "".join(f" OOD $\\tau$ ({k}) | $\\Delta$ |" for k in strata))
    rule = "|---|---|---|---|" + "---|---|" * len(strata)
    for group in dict.fromkeys(r["group"] for r in results):
        rows = [r for r in results if r["group"] == group]
        first = rows[0]["rows"][0]["width"]
        out += [f"### {group}", "", head.format(first=first), rule]
        for res in rows:
            ood = res["rows"][1:]
            base = np.mean([r["tau"] for r in ood])
            cells = [res["condition"], res["arm"],
                     f"{res['rows'][0]['tau']:+.4f}", f"{base:+.4f}"]
            for k in strata:
                ms = np.mean([r[f"tau_{k}"] for r in ood])
                cells += [f"{ms:+.4f}", f"{ms - base:+.4f}"]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["cifar10", "svhn"], required=True)
    ap.add_argument("--strata", nargs="+", choices=["lr", "hp"], default=["lr", "hp"])
    ap.add_argument("--out", type=str, default=None,
                    help="write markdown here instead of stdout")
    ap.add_argument("--compact", action="store_true",
                    help="one row per condition (in-distribution + OOD means) instead of "
                         "a per-width table each")
    args = ap.parse_args()

    root = Path(os.environ.get("ANYDIM_DATA_ROOT", "data"))
    results = []
    for group, dirname in PRED_DIRS[args.dataset]:
        d = Path(dirname)
        for npz in sorted(d.glob("preds_*.npz")):
            if condition_of(npz.stem) in OUT_OF_SCOPE:
                continue
            res = analyse(npz, args.dataset, root, args.strata)
            if res is None:
                continue
            # v2 supersedes v1 for the same (condition, arm): same checkpoint, wider ladder.
            key = (res["condition"], res["arm"], group)
            prior = next((i for i, r in enumerate(results)
                          if (r["condition"], r["arm"], r["group"]) == key), None)
            res["group"] = group
            if prior is None:
                results.append(res)
            elif len(res["rows"]) >= len(results[prior]["rows"]):
                results[prior] = res
    if not results:
        raise SystemExit(f"no prediction dumps found for {args.dataset}")
    results.sort(key=lambda r: (r["group"], r["condition"], r["arm"]))
    title = (f"Diagnostic 4 — Kendall tau within hyperparameter strata "
             f"({args.dataset.upper()})")
    renderer = render_compact if args.compact else render
    text = renderer(results, args.strata, title)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out} ({len(results)} conditions)")
    else:
        print(text)


if __name__ == "__main__":
    main()
