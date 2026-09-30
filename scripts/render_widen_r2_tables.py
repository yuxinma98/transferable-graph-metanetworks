"""Recompute per-width R^2 for Experiment 2 (hand-crafted widening) on the two CNN
accuracy-prediction datasets, from the prediction arrays `eval_sizegen_predgen_duplicated.py`
already dumped to disk (`--save-predictions`), and print the markdown table each results
doc's `### {family} widening` / `#### {ScaleGMN,Plain GMN}` subsection is missing -- it
currently records only the two-endpoint R^2 in a compact aside sentence.

No eval is rerun: this reads the (pred, actual) arrays already saved by the widening-family
eval jobs (`/tmp/predgen_{dataset}_widen_families/preds_{family}_{direction}.npz`), and
prints one markdown table per (family, model family), to be pasted by hand into the results
doc right after the existing Kendall tau table -- same convention as
`render_predgen_v2_results.py --stdout`.

Usage:
    python scripts/render_widen_r2_tables.py --dataset cifar10
    python scripts/render_widen_r2_tables.py --dataset svhn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

WIDTHS = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]

# model-family h4 heading -> [(row-label stem, npz condition label)], in the same row
# order as the existing tau tables. Bold (exact invariance) is auto-detected from the
# recomputed values rather than hardcoded.
ROWS = {
    "ScaleGMN": [
        ("ScaleGMN baseline", "ScaleGMN baseline"),
        ("ScaleGMN dup-equiv", "ScaleGMN dup-equiv"),
        ("mp-ScaleGMN dup-equiv", "mp-ScaleGMN dup-equiv"),
    ],
    "GMN": [
        ("GMN baseline", "GMN baseline"),
        ("GMN dup-equiv", "GMN dup-equiv"),
        ("mp-GMN dup-equiv", "mp-GMN dup-equiv"),
    ],
}
DIRS = {"fw": "forward", "bd": "bidirectional"}
FLAT_TOL = 1e-3


def fmt_r2(x: float) -> str:
    return f"{x:+.3f}".replace("-", "−")


def fmt_drop(x: float) -> str:
    return f"{x:.3f}" if x >= 0 else f"−{abs(x):.3f}"


def load(dataset: str, family: str, direction: str) -> np.lib.npyio.NpzFile:
    p = Path(f"/tmp/predgen_{dataset}_widen_families/preds_{family}_{direction}.npz")
    return np.load(p)


def r2_per_width(npz: np.lib.npyio.NpzFile, cond: str) -> dict[int, float]:
    out = {}
    for w in WIDTHS:
        pk, ak = f"{cond}_w{w}_pred", f"{cond}_w{w}_actual"
        if pk not in npz.files:
            continue
        out[w] = float(r2_score(npz[ak], npz[pk]))
    return out


def table_for(dataset: str, family: str, model_family: str) -> str:
    header = "| Condition | " + " | ".join(f"w{w}" for w in WIDTHS) + " | max_drop |"
    sep = "|" + "---|" * (len(WIDTHS) + 2)
    lines = [header, sep]
    for stem, cond in ROWS[model_family]:
        for d_key, d_name in DIRS.items():
            row_label = f"{stem}, {d_key}"
            if model_family == "ScaleGMN" and d_key == "bd":
                lines.append(f"| {row_label} | " + " | ".join(["—"] * len(WIDTHS))
                            + " | out of scope |")
                continue
            npz = load(dataset, family, d_name)
            vals = r2_per_width(npz, cond)
            if not vals:
                continue
            is_flat = (max(vals.values()) - min(vals.values())) < FLAT_TOL
            cells = [fmt_r2(vals[w]) for w in WIDTHS]
            drop = vals[WIDTHS[0]] - min(vals.values())
            drop_s = fmt_drop(drop)
            if is_flat:
                row_label, cells, drop_s = f"**{row_label}**", [f"**{c}**" for c in cells], f"**{drop_s}**"
            lines.append(f"| {row_label} | " + " | ".join(cells) + f" | {drop_s} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["cifar10", "svhn"], required=True)
    args = ap.parse_args()

    for family in ("general", "uniform"):
        for model_family in ("ScaleGMN", "GMN"):
            print(f"\n#### {model_family} -- {family} widening -- R^2 per width\n")
            print(table_for(args.dataset, family, model_family))
        print()


if __name__ == "__main__":
    main()
