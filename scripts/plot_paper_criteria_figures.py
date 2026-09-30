"""Build the per-setting "four criteria" figures for the accuracy-prediction appendix.

One figure per (CNN zoo, model family, direction), a 1x4 row of panels over the four
criteria the appendix defines for accuracy prediction, in the order it defines them --
from the strictest to the most forgiving:

    MAE   ->   R^2   ->   R^2_recal   ->   Kendall tau_b

so that a condition whose curves fall in the left panels but hold up in the right ones is
reading as a calibration failure, and one that falls in all four as a ranking failure.
Each panel carries the same six series as the paper's Experiment-3 figure (three variants
x two input parameterizations), so the criteria are compared at fixed everything else.

  fig_criteria_cnn_gmn_fw.pdf         CIFAR-10, plain GMN, forward
  fig_criteria_cnn_gmn_bd.pdf         CIFAR-10, plain GMN, bidirectional
  fig_criteria_cnn_scalegmn_fw.pdf    CIFAR-10, ScaleGMN, forward
  fig_criteria_svhn_gmn_fw.pdf        SVHN, plain GMN, forward
  fig_criteria_svhn_gmn_bd.pdf        SVHN, plain GMN, bidirectional
  fig_criteria_svhn_scalegmn_fw.pdf   SVHN, ScaleGMN, forward

Six, not eight: bidirectional ScaleGMN is out of scope on both zoos (the reciprocal
backward edge feature a bidirectional positive-scale-equivariant model needs overflows
float32 on CNN graphs). SVHN never ran it, and CIFAR-10's four pre-fix conditions are a
different, non-equivariant model kept only for the record -- the same reason
`plot_sizegen_results.py` skips those eight cells.

Drawing conventions follow `plot_paper_figures.py` exactly -- colour is the variant
(blue plain / magenta duplication-compatible / amber matrix-product), linestyle plus
marker fill plus alpha the input parameterization (dashed hollow translucent SP, solid
filled opaque muP), dotted grey for reference levels, one legend below the panels -- with
two additions:

  * y limits are shared across the three figures of a zoo, per criterion, and are
    *measured* from every series that zoo's figures draw (printed on every run) rather
    than hardcoded. The two zoos keep their own limits: CIFAR-10's worst $R^2$ is $-3.98$
    against SVHN's $-1.13$, and a shared axis would flatten SVHN.
  * the $\tau_b$ panel carries that zoo's hyperparameter-only reference $\tau_{HP}$, read
    from the zoo's own quality table (the two zoos disagree on it sharply), as one dotted
    grey line per input parameterization -- the two arms have their own reference and cross
    over on CIFAR-10, so a single line would misstate both.

Numbers are read out of the results markdown by `plot_sizegen_results.py`'s parsing core,
so these figures cannot drift from the recorded (wandb-backed) results.

Usage:
    python scripts/plot_paper_criteria_figures.py                    # -> paper/figures/
    python scripts/plot_paper_criteria_figures.py --outdir /tmp/f --no-pdf
    python scripts/plot_paper_criteria_figures.py --only fig_criteria_svhn_gmn_fw -v
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from plot_paper_figures import (
    ORDER,
    PARAM_ALPHA,
    PARAM_LABEL,
    PARAM_LS,
    axes_frame,
    draw_series,
    fit_box_aspect,
    legend_handles,
    paper_style,
    save,
)
from plot_sizegen_results import (
    GMN_H3,
    INK_2,
    MP_GMN_H3_CNN,
    MP_SGMN_H3_CNN,
    MUTED,
    REPO,
    SGMN_H3,
    col_getter,
    column,
    find_table,
)

#: h2 of the Experiment-3 section, identical in both CNN docs.
H2 = "Experiment 3 — Size Generalization on Our Multi-Width Zoo"

#: Left to right: strictest criterion first, as the appendix enumerates them.
#: `table` / `field` locate the cell in the results doc -- table 0 under a
#: `#### {Forward,Bidirectional}` heading is per-width $\tau$, table 1 is the
#: `R^2 / L1 / R^2_recal` triple, whose fields are split by `_num`. `mps_col` is the
#: column name in the conv matrix-product ScaleGMN section, which keeps all four
#: criteria in one wide table instead.
CRITERIA = [
    dict(key="l1", title="MAE", table=1, field=1, mps_col="L1", ref=None),
    dict(key="r2", title="$R^2$", table=1, field=0, mps_col="R²", ref="zero"),
    dict(key="r2rc", title="$R^2_{\\mathrm{recal}}$", table=1, field=2, mps_col="R²_recal",
         ref="zero"),
    dict(key="tau", title="Kendall $\\tau_b$", table=0, field=0, mps_col="τ", ref="tau_hp"),
]

#: (doc key, model, direction) per figure; ScaleGMN bidirectional is absent by scope.
FIGURES = [(doc, model, direction)
           for doc in ("cnn", "svhn")
           for model, direction in (("gmn", "fw"), ("gmn", "bd"), ("scalegmn", "fw"))]

XTICK_ROTATION = 90
XTICK_SIZE = 5.5
XTICK_PAD = 1.5
YTICK_SIZE = 6.5
PANEL_TITLE_SIZE = 7.5
LABEL_SIZE = 7.5
LEGEND_SIZE = 7.0
LW = 1.2
MS = 2.2
BOX_ASPECT = 3 / 4
WIDTH = 5.5              # inches = \textwidth, so the fonts land at nominal size
#: Dot pattern of the $\tau_{HP}$ reference line, per input parameterization. Dotted and grey
#: keeps both in the reference channel, and sparse-vs-dense repeats the SP/muP split the
#: series already carry in linestyle, marker fill and alpha. Each arm gets its own legend
#: entry, since the two references are separate levels and cross over on CIFAR-10.
TAU_HP_LS = {"sp": (0, (1, 2.5)), "mup": (0, (1, 1))}
TAU_HP_LABEL = {"sp": "$\\tau_{HP}$ (SP)", "mup": "$\\tau_{HP}$ ($\\mu$P)"}
#: Fraction of a panel's data range left as margin above and below.
YPAD = 0.06


def criteria_series(doc: str, model: str, direction: str, param: str,
                    crit: dict) -> dict[str, object]:
    """role -> getter() -> (xs, ys) for one criterion, one arm. Mirrors
    `plot_sizegen_results.get_series`' CNN branch, generalized from its two metrics to
    all four criteria."""
    p = "muP16" if param == "mup" else "SP"
    h3 = SGMN_H3 if model == "scalegmn" else GMN_H3
    h4 = "Forward" if direction == "fw" else "Bidirectional"
    idx, field = crit["table"], crit["field"]

    series = {
        "original": col_getter(doc, H2, f"{p} baseline", h3=h3, h4=h4, idx=idx,
                               field_idx=field),
        "duplication-compatible": col_getter(doc, H2, f"{p} dup-equiv", h3=h3, h4=h4,
                                             idx=idx, field_idx=field),
    }
    if model == "gmn":
        series["matrix-product"] = col_getter(doc, H2, p, h3=MP_GMN_H3_CNN, h4=h4,
                                              idx=idx, field_idx=field)
    else:
        # ScaleGMN, forward only (bidirectional is filtered out of FIGURES). Its
        # matrix-product section is one wide table, one column per (arm, criterion).
        series["matrix-product"] = col_getter(doc, H2, f"{p} {crit['mps_col']}",
                                             h3=MP_SGMN_H3_CNN)
    return series


def tau_hp(doc: str, param: str) -> tuple[list[int], list[float]]:
    """The hyperparameter-only reference $\\tau_{HP}(16, w)$ for this arm, read from the
    zoo's own quality table: ranking by the sampled hyperparameters alone, ignoring the
    weights. The two zoos disagree on it (SP bottoms out at $0.457$ on CIFAR-10, $0.381$
    on SVHN), so it is never shared between them."""
    p = "muP16" if param == "mup" else "SP"
    t = find_table(doc, "Input-network zoo quality",
                   "Baseline: ranking via hyperparameters", first_col="Width")
    return column(t, f"{p} τ_HP")


def figure_data(doc: str, model: str, direction: str) -> list[dict]:
    """Every series one figure draws, resolved: one dict per criterion, holding the
    per-arm role -> (xs, ys) maps and the widths present."""
    out = []
    for crit in CRITERIA:
        arms = {param: {role: getter() for role, getter
                        in criteria_series(doc, model, direction, param, crit).items()}
                for param in PARAM_LS}
        widths = sorted({w for series in arms.values()
                         for xs, _ in series.values() for w in xs})
        out.append(dict(crit=crit, arms=arms, widths=widths))
    return out


def measure_ylims(figures: list[tuple[str, str, str]]) -> dict[tuple[str, str], tuple[float, float]]:
    """(doc, criterion) -> y limits, from every value drawn in that zoo's figures, so a
    criterion's panel is on one axis across the zoo's three figures. Criteria with a
    natural bound keep it: MAE starts at $0$, and the three correlation-like criteria are
    not padded above $1$."""
    values: dict[tuple[str, str], list[float]] = {}
    for doc, model, direction in figures:
        for panel in figure_data(doc, model, direction):
            key = (doc, panel["crit"]["key"])
            for series in panel["arms"].values():
                for _, ys in series.values():
                    values.setdefault(key, []).extend(ys)
            if panel["crit"]["ref"] == "tau_hp":
                for param in PARAM_LS:
                    values[key].extend(tau_hp(doc, param)[1])

    lims = {}
    for (doc, key), ys in values.items():
        lo, hi = min(ys), max(ys)
        pad = YPAD * (hi - lo)
        lo, hi = (0.0 if key == "l1" else lo - pad), hi + pad
        if key != "l1":
            # All three of $\tau_b$, $R^2$ and $\Rrecal$ are bounded above by $1$, so the
            # headroom is a fixed small margin rather than a fraction of the range --
            # CIFAR-10's $R^2$ range is five units wide and would otherwise spend a
            # quarter of the panel on space no series can reach.
            hi = min(hi, 1.04)
        lims[(doc, key)] = (lo, hi)
    return lims


def draw_panel(ax, panel: dict, doc: str, ylim: tuple[float, float], verbose: bool):
    crit = panel["crit"]
    xmap = axes_frame(ax, panel["widths"], ylim, rotation=XTICK_ROTATION,
                      labelsize=XTICK_SIZE, pad=XTICK_PAD)
    ax.tick_params(axis="y", labelsize=YTICK_SIZE)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))

    if crit["ref"] == "zero":
        ax.axhline(0.0, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=0)
    elif crit["ref"] == "tau_hp":
        # One line per arm, not a band over both: each parameterization has its own
        # hyperparameter-only reference, and on CIFAR-10 the two cross over, so a single
        # level would misstate both.
        for param in PARAM_LS:
            pts = [(xmap[w], y) for w, y in zip(*tau_hp(doc, param)) if w in xmap]
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], color=MUTED, lw=1.0,
                        ls=TAU_HP_LS[param], alpha=PARAM_ALPHA[param], zorder=0)

    # muP first, SP last: SP's hollow markers would otherwise be buried under muP's
    # filled ones wherever the two nearly coincide.
    for param in ("mup", "sp"):
        draw_series(ax, panel["arms"][param], ls=PARAM_LS[param], xmap=xmap, lw=LW,
                    ms=MS, alpha=PARAM_ALPHA[param], filled=(param == "mup"))
    ax.set_title(crit["title"], pad=3, fontsize=PANEL_TITLE_SIZE, color=INK_2)

    if verbose:
        for param, series in panel["arms"].items():
            print(f"  {crit['key']:5s}/{param:3s}: " + "  ".join(
                f"{role}=" + ",".join(f"w{w}:{y:g}" for w, y in zip(*series[role]))
                for role in ORDER if role in series))


def legend(fig, model: str, refs: list[Line2D]):
    """Two rows: the three variants on the first, the two parameterization keys and the
    reference levels on the second, filled column-major."""
    models = legend_handles(ORDER, model, lw=LW, ms=MS)
    keys = [Line2D([], [], color=INK_2, lw=LW, ls=PARAM_LS[p], alpha=PARAM_ALPHA[p],
                   marker="o", ms=MS, mfc=INK_2 if p == "mup" else "white", mec=INK_2,
                   mew=1.1 if p == "mup" else 1.3, label=PARAM_LABEL[p])
            for p in PARAM_LS] + refs
    while len(models) < len(keys):
        models.append(Line2D([], [], color="none", label=" "))
    fig.legend(handles=[h for pair in zip(models, keys) for h in pair],
               loc="outside lower center", ncol=len(keys), frameon=False,
               handlelength=2.2, columnspacing=1.3, fontsize=LEGEND_SIZE)


def fig_criteria(outdir: Path, want_pdf: bool, name: str, doc: str, model: str,
                 direction: str, ylims: dict, verbose: bool):
    paper_style()
    fig = plt.figure(figsize=(WIDTH, 2.4), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0, wspace=0)
    # The panels live in a subfigure so that its `supxlabel` sits under the ticks; on the
    # parent figure it would land in the same slot as the outside legend.
    panels = fig.subfigures(1, 1)
    axs = panels.subplots(1, len(CRITERIA))
    for ax, panel in zip(axs, figure_data(doc, model, direction)):
        draw_panel(ax, panel, doc, ylims[(doc, panel["crit"]["key"])], verbose)
    panels.supxlabel("Test Width", fontsize=LABEL_SIZE)
    legend(fig, model, [
        Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 2)), label="$R^2 = 0$"),
    ] + [Line2D([], [], color=MUTED, lw=1.0, ls=TAU_HP_LS[p], alpha=PARAM_ALPHA[p],
                label=TAU_HP_LABEL[p]) for p in PARAM_LS])
    fit_box_aspect(fig, axs[-1], BOX_ASPECT)
    if verbose:
        print(f"  figure {fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in")
    save(fig, name, outdir, want_pdf)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=REPO / "paper" / "figures")
    ap.add_argument("--no-pdf", action="store_true", help="PNG only")
    ap.add_argument("--only", nargs="*", help="figure names to build (default: all)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    specs = [(f"fig_criteria_{doc}_{model}_{direction}", doc, model, direction)
             for doc, model, direction in FIGURES]
    if args.only:
        specs = [s for s in specs if s[0] in args.only]
        if not specs:
            ap.error(f"--only matched nothing; names are "
                     f"{[f'fig_criteria_{d}_{m}_{x}' for d, m, x in FIGURES]}")

    # Limits always come from the full set, so a subset build lands on the same axes as
    # the figures beside it.
    ylims = measure_ylims(FIGURES)
    for (doc, key), (lo, hi) in ylims.items():
        print(f"ylim {doc:4s} {key:5s} = [{lo:+.3f}, {hi:+.3f}]")

    for name, doc, model, direction in specs:
        print(name)
        fig_criteria(args.outdir, not args.no_pdf, name, doc, model, direction, ylims,
                     args.verbose)


if __name__ == "__main__":
    main()
