"""Paper-facing Experiment-2 (hand-crafted widening) figures.

One figure per (model family, widening family): a 1x4 row of panels grouped as
INR Classification (MNIST, FMNIST) | Generalization Prediction (CIFAR-10, SVHN),
matching `fig_sizegen`'s two-level grouped layout so the two figures read as one
family in the paper.

Unlike `fig_sizegen` (independently trained wide weights), this reads Experiment 2's
hand-crafted-widening tables via `plot_widen_families.series_for`, so a combined figure
cannot drift from the per-dataset `results/figures/widen_*.png` ones or from the results
tables they are both read out of.

  fig_widen_dup_gmn.pdf / .png         duplication widening (eq:dup),   plain GMN family
  fig_widen_block_gmn.pdf / .png       block-wise widening (eq:blockwise), plain GMN family
  fig_widen_dup_scalegmn.pdf / .png    duplication widening,            ScaleGMN family
  fig_widen_block_scalegmn.pdf / .png  block-wise widening,             ScaleGMN family

Each panel carries up to three series -- baseline, duplication-compatible,
matrix-product -- forward (solid) and bidirectional (dashed) wherever both are in scope.
Two omissions are not annotated on the (small) panels, only in the caption: mp-ScaleGMN
is forward-only in every experiment, and bidirectional ScaleGMN of any variant is out of
scope on the two CNN graphs (CIFAR-10, SVHN).

Usage:
    python scripts/plot_paper_widen_figures.py                 # -> paper/figures/
    python scripts/plot_paper_widen_figures.py --outdir /tmp/f --no-pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_paper_figures import (
    AMBER,
    axes_frame,
    fit_box_aspect,
    inches_box,
    inches_wide,
    paper_style,
    save,
)
from plot_sizegen_results import BLUE, INK_2, MAGENTA, MUTED, REPO
from plot_widen_families import DATASETS, DIR_STYLE, METRIC, PANELS, series_for

# Paper-facing series names + colours, matching fig_sizegen's PAPER_STYLE colour slots
# (blue / magenta / amber). Unlike fig_sizegen, roles here also carry distinct marker
# shapes (matching plot_widen_families.ROLE_STYLE's o/s/^): under the duplication
# widening, the duplication-compatible and matrix-product curves are both exactly flat
# and land on nearly the same value, so colour alone is not enough to tell them apart
# where they coincide -- the marker shape still is.
LABEL = {
    "original": {"gmn": "GMN", "scalegmn": "ScaleGMN"},
    "duplication-compatible": {"gmn": "Duplication-compatible GMN",
                                "scalegmn": "Duplication-compatible ScaleGMN"},
    "matrix-product": {"gmn": "Matrix-product GMN", "scalegmn": "Matrix-product ScaleGMN"},
}
ROLE_COLOR = {"original": BLUE, "duplication-compatible": MAGENTA, "matrix-product": AMBER}
ROLE_MARKER = {"original": "o", "duplication-compatible": "s", "matrix-product": "^"}
ORDER = ["original", "duplication-compatible", "matrix-product"]

#: this script's `widening` key -> index into `plot_widen_families.PANELS`.
WIDEN_KEY = {"dup": 1, "block": 0}

WIDTH = 5.5             # inches = \textwidth, so the PNG lands at nominal font sizes
#: FMNIST's panel now carries 11 widths (up to w1024) -- too many to fit horizontally at
#: any legible size, so ticks rotate sideways like fig_sizegen's do for the same reason.
XTICK_SIZE = 5.5
XTICK_ROTATION = 90
XTICK_PAD = 1.5
YTICK_SIZE = 6.5
GROUP_TITLE_SIZE = 8.5
PANEL_TITLE_SIZE = 7.5
LABEL_SIZE = 7.5
LEGEND_SIZE = 7.0
LW = 1.3
#: Matches `fig_sizegen`'s SIZEGEN_MS, so a marker is the same size in every sizegen-family
#: figure in the paper.
MS = 2.2
BOX_ASPECT = 3 / 4
#: Bidirectional (dashed) lines get a hollow marker, mirroring `fig_sizegen`'s SP/muP cue --
#: but a higher alpha than that figure's 0.55, since here the dash pattern alone already
#: carries most of the forward/bidirectional distinction and these curves still need to
#: read clearly where they diverge from the forward ones.
DIR_ALPHA = {"fw": 1.0, "bd": 0.75}


#: (widening key -> dataset -> (ylim, yticks)). Duplication (`uniform`) widening barely
#: dents either zoo's R^2 (worst -0.146 SVHN, +0.246 CIFAR-10 -- it never goes negative),
#: whereas block-wise (`general`) widening collapses both much further (-1.919 / -5.099),
#: so reusing one pair of ranges across both figures would plot the flat dup-widening
#: curves on an axis calibrated for the other experiment's much deeper collapse. Each gets
#: its own range, sized to its own worst observed value (small margin below it, or below
#: 0 if the worst value never goes negative, so the chance/$R^2=0$ reference line stays
#: visible); CIFAR-10 and SVHN still don't share one within `block`, for the reason given
#: in `fig_sizegen`'s own per-dataset R^2 limits.
CNN_YLIM = {
    "dup": {
        "cifar10": ((-0.1, 1.1), (0, 0.5, 1)),
        "svhn": ((-0.25, 1.1), (0, 0.5, 1)),
    },
    "block": {
        "cifar10": ((-2.0, 1.1), (-2, -1, 0, 1)),
        "svhn": ((-5.2, 1.1), (-5, -3, -1, 1)),
    },
}


def groups_spec(family: str, widening: str) -> list[dict]:
    h3 = PANELS[WIDEN_KEY[widening]][1]
    cols_inr = [
        dict(key="mnist", title="MNIST", h3=h3, ylabel="Test Accuracy (%)"),
        dict(key="fmnist", title="FMNIST", h3=h3),
    ]
    cifar10_ylim, cifar10_yticks = CNN_YLIM[widening]["cifar10"]
    svhn_ylim, svhn_yticks = CNN_YLIM[widening]["svhn"]
    cols_cnn = [
        dict(key="cifar10", title="CIFAR-10", h3=h3, ylabel="Test $R^2$",
             ylim=cifar10_ylim, yticks=cifar10_yticks),
        dict(key="svhn", title="SVHN", h3=h3, ylim=svhn_ylim, yticks=svhn_yticks),
    ]
    return [
        dict(title="INR Classification", sharey=True, cols=cols_inr),
        dict(title="Generalization Prediction", sharey=False, cols=cols_cnn),
    ]


def widen_panels(fig, family: str, widening: str, width_ratios: list[float],
                 xtick_pads: list[float] | None = None,
                 verbose: bool = False) -> list[tuple[object, list]]:
    pads = xtick_pads or [XTICK_PAD] * 2
    out = []
    for pad, (group, spec_group) in zip(
            pads, zip(fig.subfigures(1, 2, wspace=0.02, width_ratios=width_ratios),
                     groups_spec(family, widening))):
        group.suptitle(spec_group["title"], fontsize=GROUP_TITLE_SIZE)
        group.supxlabel("Test Width", fontsize=LABEL_SIZE)
        axs = group.subplots(1, 2, sharey=spec_group["sharey"])
        out.append((group, list(axs)))
        for ax, spec in zip(axs, spec_group["cols"]):
            _, metric_ylim, ref = METRIC[spec["key"]]
            ylim = spec.get("ylim", metric_ylim)
            data = series_for(spec["key"], family, spec["h3"])
            widths = sorted({w for xs, _ in data.values() for w in xs})
            xmap = axes_frame(ax, widths, ylim, rotation=XTICK_ROTATION,
                              labelsize=XTICK_SIZE, pad=pad)
            ax.tick_params(axis="y", labelsize=YTICK_SIZE)
            if "yticks" in spec:
                ax.set_yticks(spec["yticks"])
            ax.axhline(ref[0], color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=0)
            train_w = DATASETS[spec["key"]][2]
            if train_w in xmap:
                ax.axvline(xmap[train_w], color=MUTED, lw=0.8, alpha=0.55, zorder=0)

            # Flat (invariant) lines land on top of each other -- draw the ones that move
            # the most first, so a flat line is never hidden under a moving one.
            for (role, d), (xs, ys) in sorted(
                    data.items(), key=lambda kv: max(kv[1][1]) - min(kv[1][1]), reverse=True):
                color = ROLE_COLOR[role]
                ls, _ = DIR_STYLE[d]
                filled = d == "fw"
                ax.plot([xmap[x] for x in xs], ys, color=color, ls=ls, lw=LW,
                        alpha=DIR_ALPHA[d], marker=ROLE_MARKER[role], ms=MS,
                        mfc=color if filled else "white", mec=color,
                        mew=1.1 if filled else 1.3, zorder=3, clip_on=False)

            ax.set_title(spec["title"], pad=3, fontsize=PANEL_TITLE_SIZE, color=INK_2)
            if "ylabel" in spec:
                ax.set_ylabel(spec["ylabel"], labelpad=2, fontsize=LABEL_SIZE)
            if verbose:
                print(f"  {family}/{widening}/{spec['key']}: " + "  ".join(
                    f"{role}/{d}=" + ",".join(f"w{w}:{y:g}" for w, y in zip(*data[(role, d)]))
                    for role in ORDER for d in ("fw", "bd") if (role, d) in data))
    return out


def widen_legend(fig, family: str):
    models = [Line2D([], [], color=ROLE_COLOR[role], lw=LW, marker=ROLE_MARKER[role], ms=MS,
                     mfc=ROLE_COLOR[role], mec=ROLE_COLOR[role], mew=1.1,
                     label=LABEL[role][family]) for role in ORDER]
    keys = [Line2D([], [], color=INK_2, lw=LW, ls=DIR_STYLE[d][0], alpha=DIR_ALPHA[d],
                   marker="o", ms=MS, mfc=INK_2 if d == "fw" else "white", mec=INK_2,
                   mew=1.1 if d == "fw" else 1.3, label=label)
            for d, label in (("fw", "forward"), ("bd", "bidirectional"))]
    keys.append(Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 2)),
                       label="reference (chance / $R^2=0$)"))
    fig.legend(handles=[h for pair in zip(models, keys) for h in pair],
               loc="outside lower center", ncol=3, frameon=False,
               handlelength=2.4, columnspacing=1.6, fontsize=LEGEND_SIZE)


def fig_widen(family: str, widening: str, outdir: Path, want_pdf: bool):
    """Built twice, exactly like `plot_paper_figures.fig_sizegen`: the first pass measures
    each title group's fixed overhead (y label, ticks) so the second can size the two
    groups (`width_ratios`) to equalize all four panel widths despite the INR group
    spending its y label on the first panel only, same as the prediction group."""
    paper_style()
    ratios, pads = [1.0, 1.0], None
    for final in (False, True):
        fig = plt.figure(figsize=(WIDTH, 2.4), constrained_layout=True)
        fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0, wspace=0)
        groups = widen_panels(fig, family, widening, ratios, pads, verbose=final)
        widen_legend(fig, family)
        fit_box_aspect(fig, groups[-1][1][-1], BOX_ASPECT)
        if final:
            save(fig, f"fig_widen_{widening}_{family}", outdir, want_pdf)
            return
        overhead = [inches_wide(fig, group) - sum(inches_wide(fig, ax) for ax in axs)
                    for group, axs in groups]
        a = (WIDTH - sum(overhead)) / 4
        ratios = [2 * a + o for o in overhead]
        label_h = [max(inches_box(fig, t).height * 72 for ax in axs
                       for t in ax.get_xticklabels() if t.get_text())
                   for _, axs in groups]
        pads = [XTICK_PAD + max(label_h) - h for h in label_h]
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=REPO / "paper" / "figures")
    ap.add_argument("--no-pdf", action="store_true", help="PNG only")
    args = ap.parse_args()

    for family in ("gmn", "scalegmn"):
        for widening in ("dup", "block"):
            print(f"fig_widen_{widening}_{family}")
            fig_widen(family, widening, args.outdir, not args.no_pdf)


if __name__ == "__main__":
    main()
