"""Plot Experiment 2 -- every model, against both widening families.

One figure per symmetry family, because the two answer different questions and share no
model: the plain GMN figure is about permutation equivariance, the ScaleGMN one about
positive-scale equivariance, and putting six lines of each on one axis would make neither
readable.

  widen_gmn.png             GMN baseline / GMN dup-equiv / mp-GMN dup-equiv         (MNIST)
  widen_scalegmn.png        ScaleGMN baseline / dup-equiv / mp-ScaleGMN dup-equiv   (MNIST)
  widen_fmnist_gmn.png      the same two, on FMNIST (base width 32)
  widen_fmnist_scalegmn.png
  widen_cnn_gmn.png         the same two, on the CIFAR-10 CNN zoo (base width 16)
  widen_cnn_scalegmn.png
  widen_svhn_gmn.png        the same two, on the SVHN-GS CNN zoo (base width 16)
  widen_svhn_scalegmn.png

MNIST keeps the unqualified names it was published under; every other `--dataset` adds the
dataset to the stem rather than renaming them, using the same keys as
`plot_sizegen_results.py` (`cnn` = CIFAR-10 zoo).

The INR docs report the widened networks' test accuracy, the CNN docs their $R^2$ against
true accuracy, so the y-axis is per dataset (`METRIC`): only accuracy has a meaningful chance
line to draw, and the two CNN zoos do not share a y-limit (their collapses differ in depth).

Each figure is 1x2 panels over the two widening families, sharing a y-axis so the panels
are read against each other:

  general   the largest function-preserving widening -- an independent random block per
            (i, j) subject to the ROW condition only.
  uniform   the Kronecker widening (row AND column), a measure-zero special case of it,
            and the family the dup-equiv recipe is built for.

Every widening is function-preserving, so a line that is not flat is a model that is not
invariant to that family, never distribution shift. The pair of panels therefore isolates
the column condition: whatever moves between them needs it.

Encoding extends `plot_sizegen_results.py`'s. Colour + marker jointly carry the model, from
the same three palette slots as every other figure (baseline / dup-equiv / matrix-product),
so a model keeps its slot across the whole results directory. Line *style* carries the
direction -- solid forward, dashed bidirectional -- which the Experiment-3 figures do not
need because there direction is a separate figure. Dotted grey stays reserved for chance.
mp-ScaleGMN is forward-only in every experiment, so its bidirectional line is absent by
scope rather than pending, and the panel says so.

Each figure reads the results doc's h4 subsection for its own symmetry family, so the two
never cross-read each other's rows. There is one parameterization arm to read: at
`width == base_width` muP reduces exactly to SP, and no wide INR is ever fitted here.

Usage:
    python scripts/plot_widen_families.py                 # -> results/figures/
    python scripts/plot_widen_families.py --dataset fmnist
    python scripts/plot_widen_families.py --outdir /tmp/f
    python scripts/plot_widen_families.py --list          # dump parsed series, plot nothing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_sizegen_results import (
    GRID,
    INK,
    MUTED,
    RESULTS,
    ROLE_STYLE,
    ROLES,
    _save,
    row_series,
    style,
)

#: metric -> (y label, y limits, (reference level, its legend label) or None).
ACCURACY = ("test accuracy (%)", (5, 101), (10.0, "reference (chance / $R^2=0$)"))
#: $R^2$ of predicted against true accuracy. The results docs record only $R^2$ for
#: Experiment 2 on the two CNN zoos (Kendall $\tau_b$ was dropped in favour of it), and the
#: two zoos' collapses differ by more than a factor of two under this widening, so each
#: gets its own axis range rather than a shared one.
R2_CIFAR10 = ("test $R^2$", (-2.0, 1.1), (0.0, "reference (chance / $R^2=0$)"))
R2_SVHN = ("test $R^2$", (-5.2, 1.1), (0.0, "reference (chance / $R^2=0$)"))

#: dataset -> (results-doc key, its Experiment 2 h2 heading, base width, figure-name stem).
#: Every doc carries the same h3/h4 structure and the same row labels, so only these differ.
DATASETS = {
    "mnist": ("mnist", "Experiment 2 — Size generalization on wider equivalent networks",
              24, "widen"),
    "fmnist": ("fmnist", "Experiment 2 — Size generalization on wider equivalent networks",
               32, "widen_fmnist"),
    "cifar10": ("cnn", "Experiment 2 — Size Generalization on Wider Equivalent CNNs",
                16, "widen_cnn"),
    "svhn": ("svhn", "Experiment 2 — Size Generalization on Wider Equivalent CNNs",
             16, "widen_svhn"),
}

METRIC = {"mnist": ACCURACY, "fmnist": ACCURACY, "cifar10": R2_CIFAR10, "svhn": R2_SVHN}

#: The two CNN-zoo datasets, by this script's `--dataset` key (`plot_sizegen_results.CNN_DOCS`
#: is the same pair keyed by results-doc name). Their bidirectional `symmetry: scale` cells are
#: out of scope, so those figures carry a different absent-line note.
CNN_DATASETS = {"cifar10", "svhn"}

#: (panel title, h3 substring). `general` first: it is the result, `uniform` the reference.
PANELS = [
    ("$\\mathtt{general}$  (row condition only)", "general widening"),
    ("$\\mathtt{uniform}$  (row $+$ column)", "uniform (Kronecker) widening"),
]

#: figure key -> the h4 subsection its table lives under, plus (legend label, row-label
#: stem) per role in ROLES order. The row stem is completed with ", fw" / ", bd" -- the
#: exact row labels of the results tables.
FAMILIES = {
    "gmn": {
        "title": "plain GMN  ($\\mathrm{symmetry}$: permutation)",
        "h4": "Plain GMN",
        "original": ("GMN baseline", "GMN baseline"),
        "duplication-compatible": ("GMN dup-equiv", "GMN dup-equiv"),
        "matrix-product": ("mp-GMN dup-equiv", "mp-GMN dup-equiv"),
    },
    "scalegmn": {
        "title": "ScaleGMN  ($\\mathrm{symmetry}$: scale)",
        "h4": "ScaleGMN",
        "original": ("ScaleGMN baseline", "ScaleGMN baseline"),
        "duplication-compatible": ("ScaleGMN dup-equiv", "ScaleGMN dup-equiv"),
        "matrix-product": ("mp-ScaleGMN dup-equiv", "mp-ScaleGMN dup-equiv"),
    },
}

DIR_STYLE = {"fw": ("-", "forward"), "bd": ((0, (4.5, 1.8)), "bidirectional")}


def series_for(dataset: str, family: str, h3: str) -> dict[tuple[str, str], tuple[list, list]]:
    """(role, direction) -> (widths, accuracies). Empty rows are dropped, so a cell that is
    out of scope simply does not appear."""
    doc, h2, _, _ = DATASETS[dataset]
    out = {}
    for role in ROLES:
        _, stem = FAMILIES[family][role]
        for d in DIR_STYLE:
            xs, ys = row_series(doc, h2, h3, f"{stem}, {d}", FAMILIES[family]["h4"])
            if xs:
                out[(role, d)] = (xs, ys)
    return out


def draw(dataset: str, family: str, outdir: Path):
    _, _, train_width, stem = DATASETS[dataset]
    ylabel, ylim, ref = METRIC[dataset]
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True,
                             constrained_layout=True)

    per_panel = [series_for(dataset, family, h3) for _, h3 in PANELS]
    widths = sorted({w for p in per_panel for xs, _ in p.values() for w in xs})
    idx_of = {w: i for i, w in enumerate(widths)}

    for ax, (title, _), data in zip(axes, PANELS, per_panel):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, lw=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if ref is not None:
            ax.axhline(ref[0], color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=0)
        if train_width in idx_of:
            ax.axvline(idx_of[train_width], color=MUTED, lw=0.8, alpha=0.55, zorder=0)

        # Flat lines land on top of each other, so draw the invariant ones last: a hidden
        # line here would read as a missing condition.
        for (role, d), (xs, ys) in sorted(data.items(),
                                          key=lambda kv: max(kv[1][1]) - min(kv[1][1]),
                                          reverse=True):
            color, _, marker, filled = ROLE_STYLE[role]
            ls, _ = DIR_STYLE[d]
            ax.plot([idx_of[x] for x in xs], ys, color=color, ls=ls, lw=1.8,
                    marker=marker, ms=4.6, mfc=color if filled else "white", mew=1.4,
                    zorder=3)

        ax.set_title(title, pad=5)
        ax.set_xticks(range(len(widths)))
        ax.set_xticklabels([str(w) for w in widths])
        ax.set_xlim(-0.6, len(widths) - 0.4)
        ax.set_ylim(*ylim)
        # constrained_layout reserves space for an axes xlabel but stacks `supxlabel` into
        # the same slot as an "outside lower center" legend, so label the axes instead.
        ax.set_xlabel(f"test width  $w$   (base width {train_width})")

    axes[0].set_ylabel(ylabel)

    models, refs = [], []
    present = {role for p in per_panel for role, _ in p}
    for role in ROLES:
        if role not in present:
            continue
        color, _, marker, filled = ROLE_STYLE[role]
        models.append(Line2D([], [], color=color, lw=1.8, marker=marker, ms=4.6,
                             mfc=color if filled else "white", mew=1.4,
                             label=FAMILIES[family][role][0]))
    for d, (ls, label) in DIR_STYLE.items():
        if any((role, d) in p for p in per_panel for role in ROLES):
            refs.append(Line2D([], [], color=INK, ls=ls, lw=1.4, label=label))
    if ref is not None:
        refs.append(Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 2)), label=ref[1]))

    # `legend` fills column-major, so interleave to get models on the top row and the
    # direction / reference keys on the bottom one.
    ncol = max(len(models), len(refs))
    handles = [h for pair in zip(models, refs) for h in pair]
    handles += models[len(refs):] + refs[len(models):]
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol, frameon=False,
               handlelength=2.6, columnspacing=1.8)

    # mp-ScaleGMN has no bidirectional checkpoint in any experiment, and on CNN graphs no
    # `symmetry: scale` condition has one at all (float32 overflow in the reciprocal backward
    # edge feature). Say so on the figure rather than leaving a reader to count lines.
    absent = [FAMILIES[family][role][0] for role in ROLES
              for d in ("bd",) if role in present
              and not any((role, d) in p for p in per_panel)]
    if dataset in CNN_DATASETS and family == "scalegmn":
        absent = ["bidirectional $\\mathrm{scale}$: out of scope on CNN graphs"]
    elif absent:
        absent = [", ".join(absent) + ": no bidirectional variant"]
    if absent:
        # Bottom-left of the `uniform` panel: the only region empty in it, and clear of the
        # dotted chance line.
        axes[1].text(0.03, 0.12, absent[0],
                     transform=axes[1].transAxes, fontsize=7.5, color=MUTED,
                     ha="left", va="bottom", zorder=5,
                     bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    _save(fig, f"{stem}_{family}", outdir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="mnist")
    ap.add_argument("--outdir", type=Path, default=RESULTS / "figures")
    ap.add_argument("--list", action="store_true",
                    help="print every parsed series and exit without plotting")
    args = ap.parse_args()

    stem = DATASETS[args.dataset][3]
    for family in FAMILIES:
        if args.list:
            for title, h3 in PANELS:
                print(f"\n### {stem}_{family} / {h3}")
                for (role, d), (xs, ys) in series_for(args.dataset, family, h3).items():
                    print(f"  {role:24s} {d}  " + " ".join(f"w{w}={y:g}"
                                                           for w, y in zip(xs, ys)))
            continue
        print(f"\n{stem}_{family}")
        draw(args.dataset, family, args.outdir)


if __name__ == "__main__":
    main()
