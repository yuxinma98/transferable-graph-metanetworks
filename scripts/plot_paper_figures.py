"""Build the multi-panel figure the paper's experiments section uses.

Numbers are read out of the results markdown by the parsing core of
``plot_sizegen_results.py`` (same tables, same wandb-backed cells), so the paper
figure cannot drift from the recorded results.

  fig_sizegen.pdf               Experiment 3 (independently trained wide weights),
                                 plain GMN, forward direction: 1x4 row of panels under a
                                 two-level title (task family / dataset), three series
                                 (plain, duplication-compatible, matrix-product) x both
                                 parameterizations of the input networks. The per-width
                                 form of the main table, except that the two prediction
                                 columns plot R^2 rather than the table's Kendall tau_b:
                                 the calibration collapse is the sharper failure, and it
                                 is invisible in a rank correlation. This is the figure
                                 the main text includes; the other three (model,
                                 direction) combinations below are the appendix's.
  fig_sizegen_gmn_bd.pdf         As above, plain GMN, bidirectional.
  fig_sizegen_scalegmn_fw.pdf    As above, ScaleGMN, forward.
  fig_sizegen_scalegmn_bd.pdf    As above, ScaleGMN, bidirectional -- only the two INR
                                 classification columns: bidirectional ScaleGMN is out
                                 of scope on both CNN datasets (the reciprocal backward
                                 edge feature a bidirectional positive-scale-equivariant
                                 model needs overflows float32 on CNN graphs), so it was
                                 never run there.

Drawing conventions:
  * x is the test width; the quarter-width panels are too narrow to thin the tick
    labels, so they space the widths evenly and label every one. Panels differ in
    training width (24 / 32 / 16), so the leftmost point of each is in-distribution.
  * colour carries variant identity, from three slots of the IBM colour-blind-safe
    palette (blue / magenta / amber); every series uses the same filled-circle marker.
    Dotted grey is reserved for reference levels (chance, $R^2=0$).
  * linestyle carries the *input* parameterization (dashed SP, solid muP), so the two
    former rows collapse into one panel and each pair of arms can be read against a
    single axis.
  * one legend, below the panels; no in-figure titles beyond the task name.

Usage:
    python scripts/plot_paper_figures.py                 # -> paper/figures/
    python scripts/plot_paper_figures.py --outdir /tmp/f --no-pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import FigureBase
from matplotlib.lines import Line2D

from plot_sizegen_results import (
    BLUE,
    EXPERIMENTS,
    GRID,
    INK_2,
    MAGENTA,
    MUTED,
    REPO,
    get_series,
    style,
)

#: Amber, the fifth slot of the same IBM colour-blind-safe palette as BLUE/MAGENTA.
AMBER = "#ffb000"

# Paper-facing series names, in fixed order; the style slot is keyed off the
# results-doc role name so the two figures agree with results/figures/. Naming carries
# the model family (GMN vs ScaleGMN), keyed separately since the same three roles are
# drawn once per (model, direction) figure.
LABEL_BY_MODEL: dict[str, dict[str, str]] = {
    "gmn": {
        "original": "GMN",
        "duplication-compatible": "Duplication-compatible GMN",
        "matrix-product": "Matrix-product GMN",
    },
    "scalegmn": {
        "original": "ScaleGMN",
        "duplication-compatible": "Duplication-compatible ScaleGMN",
        "matrix-product": "Matrix-product ScaleGMN",
    },
}
PAPER_STYLE: dict[str, tuple[str, str, str, bool]] = {
    "original": (BLUE, "-", "o", True),
    "duplication-compatible": (MAGENTA, "-", "o", True),
    "matrix-product": (AMBER, "-", "o", True),
}
ORDER = ["original", "duplication-compatible", "matrix-product"]

#: fig_sizegen only: linestyle carries the parameterization the *input* networks were
#: trained under, so both arms of a variant share one colour slot in one panel.
PARAM_LS = {"sp": (0, (4, 2)), "mup": "-"}
PARAM_LABEL = {"sp": "SP inputs", "mup": "$\\mu$P inputs"}
#: SP is drawn translucent, muP fully opaque -- a second, redundant cue to the dash
#: pattern so the two are still easy to tell apart where curves nearly overlap.
PARAM_ALPHA = {"sp": 0.55, "mup": 1.0}


def paper_style():
    """Shared rcParams, then the paper-figure overrides: a panel is a third of the
    textwidth, so axis labels and ticks run smaller than in the standalone figures."""
    style()
    plt.rcParams.update({
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7.5,
    })


# --------------------------------------------------------------------------------
# panel drawing
# --------------------------------------------------------------------------------

def axes_frame(ax, widths: list[int], ylim: tuple[float, float],
               rotation: float = 0, labelsize: float | None = None,
               pad: float = 1.5) -> dict[int, float]:
    """Set up one panel's frame and x ticks; returns the width -> x-position map for
    `draw_series`. Widths are spaced evenly along x rather than by their log2 value, so
    that every one of them can be labelled -- the price is an x axis no longer
    proportional to width."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    xmap = {w: i for i, w in enumerate(widths)}
    ax.set_xlim(-0.45, len(widths) - 0.55)
    ax.set_xticks(list(xmap.values()))
    ax.set_xticklabels([str(w) for w in widths], rotation=rotation)
    ax.set_xticks([], minor=True)
    if labelsize is not None:
        ax.tick_params(axis="x", labelsize=labelsize, pad=pad)
    ax.set_ylim(*ylim)
    return xmap


def draw_series(ax, series: dict[str, tuple[list[int], list[float]]],
                ls: object | None = None, lw: float = 1.5, ms: float = 3.7,
                xmap: dict[int, float] | None = None, alpha: float = 1.0,
                filled: bool | None = None):
    """One arm of a panel. `ls` overrides the per-role linestyle, which is how
    fig_sizegen puts both parameterizations of a variant on one colour slot; `filled`
    likewise overrides the per-role marker fill, which is how fig_sizegen makes SP
    hollow-marker and muP filled-marker -- a cue that survives even where the dash
    pattern itself is too short to read (adjacent categorical x-positions are close)."""
    for role in ORDER:
        if role not in series:
            continue
        xs, ys = series[role]
        color, role_ls, marker, role_filled = PAPER_STYLE[role]
        use_filled = role_filled if filled is None else filled
        ax.plot(xs if xmap is None else [xmap[x] for x in xs], ys,
                color=color, ls=role_ls if ls is None else ls, lw=lw, alpha=alpha,
                marker=marker, ms=ms, mfc=color if use_filled else "white",
                mec=color, mew=1.1 if use_filled else 1.3,
                zorder=3, clip_on=False)


def legend_handles(roles: list[str], model: str, lw: float = 1.5,
                   ms: float = 3.7) -> list[Line2D]:
    labels = LABEL_BY_MODEL[model]
    out = []
    for role in roles:
        color, ls, marker, filled = PAPER_STYLE[role]
        out.append(Line2D([], [], color=color, ls=ls, lw=lw, marker=marker, ms=ms,
                          mfc=color if filled else "white", mew=1.1, label=labels[role]))
    return out


def inches_box(fig, artist):
    """An artist's drawn extent, in inches. Valid only after a draw."""
    bbox = artist.bbox if isinstance(artist, FigureBase) else artist.get_window_extent()
    return bbox.transformed(fig.dpi_scale_trans.inverted())


def inches_wide(fig, artist) -> float:
    return inches_box(fig, artist).width


def fit_box_aspect(fig, ax, aspect: float, iters: int = 4):
    """Grow/shrink the figure until `ax`'s drawn box has height = aspect * width.

    `Axes.set_box_aspect` would also give a 4:3 panel, but under constrained_layout it
    shrinks the axes *inside* a full-height cell, leaving a band of slack that pushes the
    titles and legend away from the panels. Everything else in the layout (titles, ticks,
    labels, legend) has a fixed height in inches, so nudging the figure height by the
    residual converges in two or three passes and leaves no slack."""
    for _ in range(iters):
        fig.canvas.draw()
        box = inches_box(fig, ax)
        w, h = fig.get_size_inches()
        residual = box.width * aspect - box.height
        if abs(residual) < 0.01:
            break
        fig.set_size_inches(w, h + residual, forward=True)


def save(fig, name: str, outdir: Path, want_pdf: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{name}.png"
    fig.savefig(png, dpi=220, facecolor="white", bbox_inches="tight")
    print(f"  -> {png.relative_to(REPO) if png.is_relative_to(REPO) else png}")
    if want_pdf:
        pdf = outdir / f"{name}.pdf"
        fig.savefig(pdf, facecolor="white", bbox_inches="tight")
        print(f"  -> {pdf.relative_to(REPO) if pdf.is_relative_to(REPO) else pdf}")
    plt.close(fig)


# --------------------------------------------------------------------------------
# figure: independently trained wide weights
# --------------------------------------------------------------------------------

# Two-level title: a group header per task family, a dataset subheader per panel. The
# INR pair shares a y axis (identical limits); the two prediction panels do not -- SVHN's
# worst R^2 is -1.11 against CIFAR-10's -3.98, and a shared axis would flatten SVHN.
SIZEGEN_GROUPS = [
    dict(title="INR Classification", sharey=True, cols=[
        dict(key="mnist", title="MNIST", ylabel="Test Accuracy (%)",
             ylim=(0, 100), yticks=(0, 25, 50, 75, 100), ref=10.0),
        dict(key="fmnist", title="FMNIST", ylim=(0, 100), ref=10.0),
    ]),
    dict(title="Generalization Prediction", sharey=False, cols=[
        dict(key="cnn_r2", title="CIFAR-10", ylabel="Test $R^2$",
             ylim=(-4.15, 1.1), yticks=(-3, -1, 1), ref=0.0),
        # Both prediction panels are $R^2$, so one label per group -- but they keep their
        # own ticks, since the two zoos' collapses differ by a factor of four.
        dict(key="svhn_r2", title="SVHN", ylim=(-1.3, 1.1), yticks=(-1, 0, 1), ref=0.0),
    ]),
]
#: Bidirectional ScaleGMN is out of scope on both CNN datasets (module docstring), so
#: that one figure drops the "Generalization Prediction" group rather than draw two
#: panels with nothing in them.
SIZEGEN_GROUPS_INR_ONLY = [SIZEGEN_GROUPS[0]]
#: Every measured width is labelled, which is only legible on an evenly spaced x axis: on
#: log2, MNIST's tightest pair (80, 96) is 0.26 octaves = 0.047 in apart in a
#: quarter-textwidth panel, so sideways labels there overlap by 1.6 pt at 4.8 pt type and
#: clear by 0.1 pt at 4.5 pt. Evenly spaced, the 13 gaps are ~0.08 in and 5.5 pt fits.
#: Each axes box is 4:3 (width:height), fitted by `fit_box_aspect`.
SIZEGEN_XTICK_ROTATION = 90
SIZEGEN_XTICK_SIZE = 5.5
SIZEGEN_XTICK_PAD = 1.5
SIZEGEN_YTICK_SIZE = 6.5
SIZEGEN_GROUP_TITLE_SIZE = 8.5
SIZEGEN_PANEL_TITLE_SIZE = 7.5
SIZEGEN_LABEL_SIZE = 7.5
SIZEGEN_LEGEND_SIZE = 7.0
#: SP vs muP is carried primarily by marker fill (hollow vs filled, set in
#: `sizegen_panels`), reinforced by dash pattern and alpha -- markers need to be large
#: enough for hollow-vs-filled to read, even packed 13 to a curve.
SIZEGEN_LW = 1.2
SIZEGEN_MS = 2.2
SIZEGEN_BOX_ASPECT = 3 / 4
SIZEGEN_WIDTH = 5.5            # inches = \textwidth, so the fonts land at nominal size
#: Reference column count (both groups of SIZEGEN_GROUPS): the panel width that
#: SIZEGEN_WIDTH/SIZEGEN_FULL_COLS works out to. fig_sizegen() scales its own figure width
#: down proportionally to `total_cols` (e.g. SIZEGEN_GROUPS_INR_ONLY's 2 columns get
#: SIZEGEN_WIDTH/2 inches, not the full SIZEGEN_WIDTH), so every sizegen figure -- whatever
#: subset of groups it draws -- gets the *same* physical panel width, and hence the same
#: font-to-panel ratio, as the main 1x4 figure. The corresponding `\includegraphics` width
#: in LaTeX must then be `(total_cols/SIZEGEN_FULL_COLS)\textwidth`, not `\textwidth`, so
#: the PNG/PDF is still placed 1:1 and fonts land at nominal points (module docstring).
SIZEGEN_FULL_COLS = sum(len(g["cols"]) for g in SIZEGEN_GROUPS)


def sizegen_panels(fig, groups: list[dict], model: str, direction: str,
                   width_ratios: list[float], xtick_pads: list[float] | None = None,
                   verbose: bool = False) -> tuple[list[tuple[object, list]], set[str]]:
    """Draw `groups`' panels into `fig`, one titled subfigure per task family. Returns
    ((subfigure, its axes) per group, the set of roles actually drawn anywhere) so the
    caller can measure the layout and build a legend that matches what is on the page --
    e.g. ScaleGMN bidirectional never has a matrix-product role (out of scope by design,
    forward-only), and `groups` itself may drop the CNN group entirely for that combo."""
    exp_of = {e.key: e for e in EXPERIMENTS}
    pads = xtick_pads or [SIZEGEN_XTICK_PAD] * len(groups)
    out = []
    roles_present: set[str] = set()
    subfigs = fig.subfigures(1, len(groups), wspace=0.02, width_ratios=width_ratios,
                             squeeze=False)[0]
    for pad, (group, spec_group) in zip(pads, zip(subfigs, groups)):
        group.suptitle(spec_group["title"], fontsize=SIZEGEN_GROUP_TITLE_SIZE)
        group.supxlabel("Test Width", fontsize=SIZEGEN_LABEL_SIZE)
        axs = group.subplots(1, 2, sharey=spec_group["sharey"])
        out.append((group, list(axs)))
        for ax, spec in zip(axs, spec_group["cols"]):
            arms = {param: {role: getter() for role, getter
                            in get_series(exp_of[spec["key"]], model, direction, param).items()}
                    for param in PARAM_LS}
            for series in arms.values():
                roles_present.update(series.keys())
            widths = sorted({w for series in arms.values()
                             for xs, _ in series.values() for w in xs})
            xmap = axes_frame(ax, widths, spec["ylim"], rotation=SIZEGEN_XTICK_ROTATION,
                              labelsize=SIZEGEN_XTICK_SIZE, pad=pad)
            ax.tick_params(axis="y", labelsize=SIZEGEN_YTICK_SIZE)
            if "yticks" in spec:
                ax.set_yticks(spec["yticks"])
            ax.axhline(spec["ref"], color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=0)
            # muP first, SP last: SP's hollow markers would otherwise get buried under
            # muP's filled ones wherever the two nearly coincide.
            for param in ("mup", "sp"):
                draw_series(ax, arms[param], ls=PARAM_LS[param], xmap=xmap,
                            lw=SIZEGEN_LW, ms=SIZEGEN_MS, alpha=PARAM_ALPHA[param],
                            filled=(param == "mup"))
            ax.set_title(spec["title"], pad=3, fontsize=SIZEGEN_PANEL_TITLE_SIZE,
                         color=INK_2)
            if "ylabel" in spec:
                ax.set_ylabel(spec["ylabel"], labelpad=2,
                              fontsize=SIZEGEN_LABEL_SIZE)
            if verbose:
                for param, series in arms.items():
                    print(f"  {param}/{spec['key']}: " + "  ".join(
                        f"{role}=" + ",".join(f"w{w}:{y:g}" for w, y in zip(*series[role]))
                        for role in ORDER if role in series))
    return out, roles_present


def sizegen_legend(fig, roles: list[str], model: str):
    """Column-major fill: the models on the first row, the two linestyle keys plus the
    reference level on the second. `roles` may be shorter than `ORDER` (ScaleGMN
    bidirectional has no matrix-product role at all); padded with an invisible handle so
    the reference-level key still lands in the second row rather than sliding up."""
    models = legend_handles(roles, model, lw=SIZEGEN_LW, ms=SIZEGEN_MS)
    while len(models) < len(ORDER):
        models.append(Line2D([], [], color="none", label=" "))
    keys = [Line2D([], [], color=INK_2, lw=SIZEGEN_LW, ls=PARAM_LS[p], alpha=PARAM_ALPHA[p],
                   marker="o", ms=SIZEGEN_MS, mfc=INK_2 if p == "mup" else "white",
                   mec=INK_2, mew=1.1 if p == "mup" else 1.3,
                   label=PARAM_LABEL[p]) for p in PARAM_LS]
    keys.append(Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 2)),
                       label="reference (chance / $R^2=0$)"))
    fig.legend(handles=[h for pair in zip(models, keys) for h in pair],
               loc="outside lower center", ncol=3, frameon=False,
               handlelength=2.4, columnspacing=1.6, fontsize=SIZEGEN_LEGEND_SIZE)


def fig_sizegen(outdir: Path, want_pdf: bool, name: str, model: str, direction: str,
                groups: list[dict]):
    """Built twice, because the title groups do not cost the same in either direction: the
    INR group spends one y label on two panels where the prediction group spends two, and
    its sideways tick labels are a digit taller (`1024` against `512`). Left alone that
    makes the INR panels wider but shorter. The first pass measures both overheads -- both
    fixed in inches, whatever the subfigure gets -- and the second sizes the groups
    (`width_ratios`) and pads the shorter labels so every panel comes out identical.
    `groups` may be just the INR group (ScaleGMN bidirectional), in which case there is
    only one subfigure to size and no width_ratios decision to make."""
    paper_style()
    total_cols = sum(len(g["cols"]) for g in groups)
    # Scale the canvas down with the column count so a panel is the same physical size
    # (hence the same font-to-panel ratio) whether it is one of four or one of two --
    # see SIZEGEN_FULL_COLS. The matching `\includegraphics` width in LaTeX is then
    # `(total_cols/SIZEGEN_FULL_COLS)\textwidth`, not `\textwidth`.
    width = SIZEGEN_WIDTH * total_cols / SIZEGEN_FULL_COLS
    ratios, pads = [1.0] * len(groups), None
    for final in (False, True):
        fig = plt.figure(figsize=(width, 2.4), constrained_layout=True)
        fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0, wspace=0)
        panel_groups, roles_present = sizegen_panels(fig, groups, model, direction, ratios,
                                                     pads, verbose=final)
        roles = [r for r in ORDER if r in roles_present]
        sizegen_legend(fig, roles, model)
        fit_box_aspect(fig, panel_groups[-1][1][-1], SIZEGEN_BOX_ASPECT)
        if final:
            save(fig, name, outdir, want_pdf)
            return
        # overhead_g = subfigure width - its two axes' widths; solve for the common panel
        # width a from total_cols*a + sum(overhead) = width.
        overhead = [inches_wide(fig, group) - sum(inches_wide(fig, ax) for ax in axs)
                    for group, axs in panel_groups]
        a = (width - sum(overhead)) / total_cols
        ratios = [len(g["cols"]) * a + o for g, o in zip(groups, overhead)]
        # Rotated tick labels stand as tall as they are long, so pad the groups whose
        # widths are written with fewer digits up to the tallest group's label band.
        label_h = [max(inches_box(fig, t).height * 72 for ax in axs
                       for t in ax.get_xticklabels() if t.get_text())
                   for _, axs in panel_groups]
        pads = [SIZEGEN_XTICK_PAD + max(label_h) - h for h in label_h]
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=REPO / "paper" / "figures")
    ap.add_argument("--no-pdf", action="store_true", help="PNG only")
    args = ap.parse_args()

    for name, model, direction, groups in [
        ("fig_sizegen", "gmn", "fw", SIZEGEN_GROUPS),
        ("fig_sizegen_gmn_bd", "gmn", "bd", SIZEGEN_GROUPS),
        ("fig_sizegen_scalegmn_fw", "scalegmn", "fw", SIZEGEN_GROUPS),
        ("fig_sizegen_scalegmn_bd", "scalegmn", "bd", SIZEGEN_GROUPS_INR_ONLY),
    ]:
        print(name)
        fig_sizegen(args.outdir, not args.no_pdf, name, model, direction, groups)


if __name__ == "__main__":
    main()
