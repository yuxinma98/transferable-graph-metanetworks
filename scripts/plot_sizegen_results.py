"""Plot test performance vs test width for the Experiment-3 size-generalization runs.

The numbers are *read out of the results markdown* rather than duplicated here, so
the figures stay in sync as pending columns land:

  results/EXPERIMENTS_MNIST_INR_classification.md
  results/EXPERIMENTS_FMNIST_INR_classification.md
  results/EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md
  results/EXPERIMENTS_SVHN_CNN_accuracy_prediction.md

Each per-width table is located by its heading path (h2 / h3 / h4) plus an occurrence
index, and a series is a named column (tables whose first header is `Width`). Cells
that are empty or `*pending*` become NaN and are dropped, so a half-filled table plots
the part that exists.

One figure = one (dataset/metric, parameterization, model family, direction) cell, so
each metanetwork variant's OOD behaviour is legible on its own axes rather than buried
in a shared-axis panel grid:

  6 experiments  x  2 parameterizations (muP, SP)  x  2 model families (Plain GMN,
  ScaleGMN)  x  2 directions (forward, bidirectional)  =  48 combinations, minus the
  8 ScaleGMN-bidirectional CNN ones (both zoos: out of scope, never generated -- the
  elementwise-division backward messages NaN under positive-scale symmetry)  =  40
  figures

Each figure carries 3 series (legend, in this order):
  original                 = baseline   (aggregator: add, raw weights, sum-pooling)
  duplication-compatible   = dup-equiv  (aggregator: mean, fan-in rescaled edges,
                                          LayerWiseMeanReadout)
  matrix-product           = mpgmn / mpsgmn (dup-equiv only; ScaleGMN's variant is
                                          forward-only by scope, so silently absent
                                          from the ScaleGMN-bidirectional figures)

Encoding, held fixed across every figure so panels are comparable: colour+marker jointly
carry variant identity (never colour alone), from three slots of the IBM colour-blind-safe
categorical palette (blue, orange, magenta). At most three series share a panel, so every
model line is solid; dotted grey is reserved for reference levels (chance, $R^2=0$,
$\tau_{HP}$).
x-axis: evenly spaced categorical ticks over the widths actually plotted in that
figure (not log-scaled by numeric value). No in-figure title/caption text -- these
are meant to sit under a LaTeX subfigure caption, so identification is by filename.

Usage:
    python scripts/plot_sizegen_results.py                 # all figures -> results/figures/
    python scripts/plot_sizegen_results.py --outdir /tmp/f
    python scripts/plot_sizegen_results.py --list          # dump what was parsed, plot nothing
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

DOCS = {
    "mnist": RESULTS / "EXPERIMENTS_MNIST_INR_classification.md",
    "fmnist": RESULTS / "EXPERIMENTS_FMNIST_INR_classification.md",
    "cnn": RESULTS / "EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md",
    "svhn": RESULTS / "EXPERIMENTS_SVHN_CNN_accuracy_prediction.md",
}
#: The two CNN-zoo docs. They share a table layout (`| Width | SP baseline | ... |` for
#: the two model families, `| Width | SP | muP16 |` for the conv matrix-product GMN)
#: that the INR docs do not, so table lookup branches on membership here.
CNN_DOCS = {"cnn", "svhn"}

# --- palette: IBM colour-blind-safe categorical (slots 1, 4, 3) -------------------
# https://lospec.com/palette-list/ibm-color-blind-safe -- the three chosen slots are
# separated in hue *and* lightness, so they survive CVD simulation and grey printing.
BLUE = "#648fff"
ORANGE = "#fe6100"
MAGENTA = "#dc267f"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"

# role -> (colour, linestyle, marker, marker-filled). At most three model series share
# a panel, so all of them are solid and identity is carried by colour + marker.
ROLE_STYLE: dict[str, tuple[str, str, str, bool]] = {
    "original": (BLUE, "-", "o", False),
    "duplication-compatible": (ORANGE, "-", "s", True),
    "matrix-product": (MAGENTA, "-", "^", True),
}
ROLES = ["original", "duplication-compatible", "matrix-product"]


# =============================================================================
# markdown table extraction (unchanged parsing core)
# =============================================================================

@dataclass
class Table:
    h2: str | None
    h3: str | None
    h4: str | None
    header: list[str]
    rows: list[list[str | None]]

    @property
    def path(self) -> str:
        return " / ".join(x for x in (self.h2, self.h3, self.h4) if x)


def _clean(cell: str) -> str | None:
    """Strip markdown emphasis / code ticks / links; empty and pending -> None."""
    s = cell.strip()
    s = s.replace("**", "").replace("`", "")
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)   # [text](url) -> text
    s = s.replace("\\", "").strip()
    if s in ("", "-", "—", "–") or s.startswith("*pending*") or s.startswith("*paused"):
        return None
    return s


def parse_tables(path: Path) -> list[Table]:
    """Every pipe table in the file, tagged with the heading path above it."""
    h2 = h3 = h4 = None
    tables: list[Table] = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(#{2,4})\s+(.*)$", line)
        if m:
            # Drop code ticks so specs can be written as plain text.
            level, title = len(m.group(1)), m.group(2).strip().replace("`", "")
            if level == 2:
                h2, h3, h4 = title, None, None
            elif level == 3:
                h3, h4 = title, None
            else:
                h4 = title
            i += 1
            continue
        if line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            if len(block) >= 2 and set(block[1].replace("|", "").strip()) <= set("-: "):
                header = [_clean(c) or "" for c in block[0].strip().strip("|").split("|")]
                rows = [[_clean(c) for c in r.strip().strip("|").split("|")] for r in block[2:]]
                tables.append(Table(h2, h3, h4, header, rows))
            continue
        i += 1
    return tables


DOC_TABLES: dict[str, list[Table]] = {}


def tables_for(doc: str) -> list[Table]:
    if doc not in DOC_TABLES:
        DOC_TABLES[doc] = parse_tables(DOCS[doc])
    return DOC_TABLES[doc]


def _matches(t: Table, h2: str | None, h3: str | None, h4: str | None) -> bool:
    for want, got in ((h2, t.h2), (h3, t.h3), (h4, t.h4)):
        if want is not None and (got is None or want not in got):
            return False
    # An omitted h4 means "not inside a deeper subsection" -- that is what
    # disambiguates the summary table from the per-direction ones.
    if h4 is None and t.h4 is not None:
        return False
    return True


def find_table(doc: str, h2: str, h3: str | None = None, h4: str | None = None,
               idx: int = 0, first_col: str | None = None) -> Table:
    hits = [t for t in tables_for(doc)
            if _matches(t, h2, h3, h4)
            and (first_col is None or t.header[0] == first_col)]
    if len(hits) <= idx:
        raise LookupError(
            f"{doc}: no table #{idx} under h2~{h2!r} h3~{h3!r} h4~{h4!r} "
            f"(first_col={first_col!r}); found {len(hits)}"
        )
    return hits[idx]


# --- cell -> number ----------------------------------------------------------

def _num(cell: str | None, field_idx: int = 0) -> float:
    """`97.1%` -> 97.1, `0.9041` -> 0.9041, `+0.973 / 0.0123 / +0.977` -> field."""
    if cell is None:
        return math.nan
    parts = [p.strip() for p in cell.split("/")]
    if field_idx >= len(parts):
        return math.nan
    # The docs write negatives with a Unicode minus (U+2212), e.g. "−2.549".
    s = parts[field_idx].replace("−", "-")
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        return math.nan
    return float(m.group(0))


def _width(cell: str | None) -> int | None:
    if cell is None:
        return None
    m = re.match(r"^w(\d+)", cell)
    return int(m.group(1)) if m else None


def column(t: Table, col: str, field_idx: int = 0) -> tuple[list[int], list[float]]:
    """Series from a `| Width | ... |` table: one column, one point per width row."""
    if col not in t.header:
        raise LookupError(f"column {col!r} not in {t.header} ({t.path})")
    j = t.header.index(col)
    xs, ys = [], []
    for row in t.rows:
        w = _width(row[0])
        if w is None or j >= len(row):
            continue
        y = _num(row[j], field_idx)
        if not math.isnan(y):
            xs.append(w)
            ys.append(y)
    return xs, ys


def col_getter(doc, h2, col, h3=None, h4=None, idx=0, field_idx=0):
    return lambda: column(find_table(doc, h2, h3, h4, idx, "Width"), col, field_idx)


def row_series(doc: str, h2: str, h3: str | None, row_label: str, h4: str | None = None,
               idx: int = 0) -> tuple[list[int], list[float]]:
    """Series from a `| Condition | w24 | ... |` table: one row, widths across the header.

    The widened-weight tables (Experiment 2 on every doc) are transposed relative to the
    Experiment-3 ones, so they need this rather than `column`. A row that exists but is
    entirely empty -- e.g. mp-ScaleGMN bidirectional, which is out of scope, not pending --
    returns two empty lists rather than raising."""
    t = find_table(doc, h2, h3, h4, idx, "Condition")
    for row in t.rows:
        if row[0] != row_label:
            continue
        xs, ys = [], []
        for j, head in enumerate(t.header):
            w = _width(head)
            if w is None or j >= len(row):
                continue
            y = _num(row[j])
            if not math.isnan(y):
                xs.append(w)
                ys.append(y)
        return xs, ys
    raise LookupError(f"{doc}: no row {row_label!r} in {t.path}")


# =============================================================================
# per-figure data spec
# =============================================================================

GMN_H3 = "Plain GMN Results (symmetry: permutation, forward and bidirectional)"
SGMN_H3 = "ScaleGMN Results (symmetry: scale, forward and bidirectional)"
MP_GMN_H3_INR = "Matrix-Product GMN Results"
MP_SGMN_H3_INR = "Matrix-Product ScaleGMN Results"
MP_GMN_H3_CNN = "Conv Matrix-Product GMN Results"
MP_SGMN_H3_CNN = "Conv Matrix-Product ScaleGMN Results"


@dataclass
class Experiment:
    key: str            # figure-name component
    doc: str
    h2: str
    mup: str            # e.g. "muP24"
    train_width: int
    ylabel: str
    ylim: tuple[float, float]
    chance: float | None = None
    metric: str | None = None      # "tau" | "r2", CNN only
    band: str | None = None        # "tau_hp" | "zero" | None


EXPERIMENTS = [
    Experiment("mnist", "mnist", "Experiment 3 — Size Generalization", "muP24", 24,
               "test accuracy (%)", (5, 101), chance=10.0),
    Experiment("fmnist", "fmnist", "Experiment 3 — Size Generalization", "muP32", 32,
               "test accuracy (%)", (5, 101), chance=10.0),
    Experiment("cnn_tau", "cnn", "Experiment 3 — Size Generalization on Our Multi-Width Zoo",
               "muP16", 16, "test Kendall $\\tau$", (0.45, 0.95), metric="tau", band="tau_hp"),
    Experiment("cnn_r2", "cnn", "Experiment 3 — Size Generalization on Our Multi-Width Zoo",
               "muP16", 16, "test $R^2$", (-4.15, 1.1), metric="r2", band="zero"),
    # SVHN's tau floor is lower than CIFAR-10's and its tau_HP reference falls to 0.3812, so
    # the tau axis starts below 0 rather than at 0.45. It has to go *negative*: plain GMN SP
    # baseline, bidirectional, is the only condition in either study to invert its ranking
    # (-0.0241 at w256 to -0.1500 at w512), and a floor of 0 silently clips it off the panel.
    # Its worst R^2 is -1.238, a third of CIFAR-10's -3.983.
    Experiment("svhn_tau", "svhn", "Experiment 3 — Size Generalization on Our Multi-Width Zoo",
               "muP16", 16, "test Kendall $\\tau$", (-0.2, 0.95), metric="tau", band="tau_hp"),
    Experiment("svhn_r2", "svhn", "Experiment 3 — Size Generalization on Our Multi-Width Zoo",
               "muP16", 16, "test $R^2$", (-1.3, 1.1), metric="r2", band="zero"),
]

DIRECTIONS = ["fw", "bd"]
MODELS = ["gmn", "scalegmn"]
PARAMS = ["mup", "sp"]


def get_series(exp: Experiment, model: str, direction: str, param: str) -> dict[str, object]:
    """role -> getter() -> (xs, ys). A role absent from the dict is out of scope for this
    arm (e.g. ScaleGMN's matrix-product variant is forward-only) rather than pending."""
    p = exp.mup if param == "mup" else "SP"
    h3 = SGMN_H3 if model == "scalegmn" else GMN_H3
    h4 = "Forward" if direction == "fw" else "Bidirectional"

    if exp.doc in CNN_DOCS:
        tbl_idx, field_idx = (0, 0) if exp.metric == "tau" else (1, 0)
        series = {
            "original": col_getter(exp.doc, exp.h2, f"{p} baseline", h3=h3, h4=h4,
                                    idx=tbl_idx, field_idx=field_idx),
            "duplication-compatible": col_getter(exp.doc, exp.h2, f"{p} dup-equiv", h3=h3, h4=h4,
                                                  idx=tbl_idx, field_idx=field_idx),
        }
        if model == "gmn":
            series["matrix-product"] = col_getter(exp.doc, exp.h2, p, h3=MP_GMN_H3_CNN, h4=h4,
                                                    idx=tbl_idx, field_idx=field_idx)
            return series
        # model == "scalegmn"; direction == "fw" -- bidirectional is filtered out upstream
        # (invalid pre-reciprocal-fix run, out of scope, not generated at all).
        col = f"{p} τ" if exp.metric == "tau" else f"{p} R²"
        series["matrix-product"] = col_getter(exp.doc, exp.h2, col, h3=MP_SGMN_H3_CNN)
        return series

    # MNIST / FMNIST
    series = {
        "original": col_getter(exp.doc, exp.h2, f"{p} baseline", h3=h3, h4=h4),
        "duplication-compatible": col_getter(exp.doc, exp.h2, f"{p} dup-equiv", h3=h3, h4=h4),
    }
    if model == "gmn":
        suffix = "forward" if direction == "fw" else "bidir"
        series["matrix-product"] = col_getter(exp.doc, exp.h2, f"{p} {suffix}", h3=MP_GMN_H3_INR)
    elif direction == "fw":
        series["matrix-product"] = col_getter(exp.doc, exp.h2, p, h3=MP_SGMN_H3_INR)
    return series


# =============================================================================
# rendering
# =============================================================================

def style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.8,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelsize": 12,
        "axes.titlesize": 9.5,
        "legend.fontsize": 8.5,
        "font.size": 9,
        "figure.dpi": 110,
        # Visually match the paper (`\usepackage{times}`): STIX is metrically
        # designed to pair with Times, and ships with matplotlib's font cache.
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
    })


def tau_hp_line(doc: str, param: str):
    """Dotted grey line = the hyperparameter-only ceiling tau_HP for this arm.

    Read from the zoo's own doc: the two CNN zoos disagree sharply on it (SP falls to
    0.457 on CIFAR-10, 0.3812 on SVHN, and only CIFAR-10's arms cross over), so this
    reference is per-dataset."""
    p = "muP16" if param == "mup" else "SP"
    t = find_table(doc, "Input-network zoo quality", "Baseline: ranking via hyperparameters",
                   first_col="Width")
    xs, ys = column(t, f"{p} τ_HP")
    return xs, ys, Line2D([], [], color=MUTED, lw=1.1, ls=(0, (1, 1)),
                          label="HP-only ceiling $\\tau_{HP}$")


def draw_single(name: str, ylabel: str, series: dict[str, object],
                 train_width: int, ylim: tuple[float, float],
                 chance: float | None, band: str | None, param: str, doc: str,
                 outdir: Path, verbose: bool):
    style()
    fig, ax = plt.subplots(figsize=(3.7, 3.15), constrained_layout=True)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    widths_here: set[int] = set()
    per_role_xy: dict[str, tuple[list[int], list[float]]] = {}
    missing: list[str] = []
    for role in ROLES:
        getter = series.get(role)
        if getter is None:
            continue
        try:
            xs, ys = getter()
        except LookupError as e:
            print(f"  ! {name} / {role}: {e}")
            missing.append(role)
            continue
        if not xs:
            missing.append(role)
            continue
        per_role_xy[role] = (xs, ys)
        widths_here.update(xs)

    if not widths_here:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.grid(False)
        msg = ("pending: " + ", ".join(missing)) if missing else "not run yet"
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color=MUTED, wrap=True)
        _save(fig, name, outdir)
        return

    widths = sorted(widths_here)
    idx_of = {w: i for i, w in enumerate(widths)}

    handles = []
    for role in ROLES:
        if role not in per_role_xy:
            continue
        xs, ys = per_role_xy[role]
        color, ls, marker, filled = ROLE_STYLE[role]
        xi = [idx_of[x] for x in xs]
        line, = ax.plot(xi, ys, color=color, ls=ls, lw=1.8, marker=marker, ms=4.6,
                        mfc=color if filled else "white", mew=1.4, zorder=3)
        handles.append(Line2D([], [], color=color, ls=ls, lw=1.8, marker=marker, ms=4.6,
                              mfc=color if filled else "white", mew=1.4, label=role))
        if verbose:
            print(f"  {name:44s} {role:24s} " + " ".join(f"w{w}:{y:g}" for w, y in zip(xs, ys)))

    if band == "tau_hp":
        hxs, hys, h = tau_hp_line(doc, param)
        hi = [idx_of[w] for w in hxs if w in idx_of]
        hy = [y for w, y in zip(hxs, hys) if w in idx_of]
        if hi:
            ax.plot(hi, hy, color=MUTED, lw=1.1, ls=(0, (1, 1)), zorder=1)
            handles.append(h)
    elif band == "zero":
        ax.axhline(0.0, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=0)
        handles.append(Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 2)),
                              label="reference (chance / $R^2=0$)"))

    if chance is not None:
        ax.axhline(chance, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=0)
        handles.append(Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 2)),
                              label="reference (chance / $R^2=0$)"))

    if train_width in idx_of:
        ax.axvline(idx_of[train_width], color=MUTED, lw=0.8, alpha=0.55, zorder=0)

    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([str(w) for w in widths], rotation=0)
    ax.set_xlim(-0.6, len(widths) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"test width  $w$   (train width {train_width})")
    ax.set_ylabel(ylabel)

    if missing:
        ax.text(0.03, 0.03, "pending: " + ", ".join(missing), transform=ax.transAxes,
                fontsize=7, color=MUTED, ha="left", va="bottom", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 2),
               frameon=False)

    _save(fig, name, outdir)


def _save(fig, name: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{name}.png"
    fig.savefig(png, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"  -> {png.relative_to(REPO) if png.is_relative_to(REPO) else png}")
    plt.close(fig)


def all_figures():
    for exp in EXPERIMENTS:
        for param_key in PARAMS:
            for model_key in MODELS:
                for dir_key in DIRECTIONS:
                    if exp.doc in CNN_DOCS and model_key == "scalegmn" and dir_key == "bd":
                        continue  # out of scope on both zoos, never run: see module docstring
                    name = f"sizegen_{exp.key}_{param_key}_{model_key}_{dir_key}"
                    series = get_series(exp, model_key, dir_key, param_key)
                    yield dict(name=name, ylabel=exp.ylabel, series=series,
                              train_width=exp.train_width, ylim=exp.ylim, chance=exp.chance,
                              band=exp.band, param=param_key, doc=exp.doc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=RESULTS / "figures")
    ap.add_argument("--only", nargs="*", help="figure names to build (default: all)")
    ap.add_argument("--list", action="store_true",
                    help="print every parsed series and exit without plotting")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    specs = list(all_figures())
    if args.only:
        specs = [s for s in specs if s["name"] in args.only]
        if not specs:
            ap.error(f"--only matched nothing; names look like: {[s['name'] for s in list(all_figures())[:4]]} ...")

    if args.list:
        for s in specs:
            print(f"\n### {s['name']}")
            for role in ROLES:
                getter = s["series"].get(role)
                if getter is None:
                    continue
                try:
                    xs, ys = getter()
                    cells = " ".join(f"w{w}={y:g}" for w, y in zip(xs, ys))
                except LookupError as e:
                    cells = f"MISSING ({e})"
                print(f"  {role:24s} {cells}")
        return

    for s in specs:
        print(f"\n{s['name']}")
        draw_single(s["name"], s["ylabel"], s["series"], s["train_width"], s["ylim"],
                   s["chance"], s["band"], s["param"], s["doc"], args.outdir, args.verbose)


if __name__ == "__main__":
    main()
