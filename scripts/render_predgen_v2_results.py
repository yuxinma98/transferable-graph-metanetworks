#!/usr/bin/env python
"""Render the CNN Stage 3 v2 re-score results into the results doc, in place.

The ten conditions of `run_cifar10_predgen_sizegen_v2.sh` are eval-only re-scores that land
one at a time in /tmp/predgen_sizegen_v2/SUMMARY.md. This script is the mechanical half of
writing them up: it parses that file and replaces the marked block in
results/EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md with

  * the cross-condition "at a glance" table (one row per finished condition), and
  * every condition's full per-width tau / tau-over-ceiling / R2 / L1 / R2_recal table.

`tau` is reported both raw and as `tau / tau_b_max(w, arm)`, the fraction of the *attainable*
ranking, because tied labels cap tau_b below 1 by an amount that is a property of each width's
label distribution rather than of the metanetwork (see `summarize_cnn_zoo_quality.tau_b_ceiling`).
On this CIFAR-10 zoo the ceiling is 0.9925-0.9997, so the normalized column is nearly a no-op and
is here mainly for auditability; on the SVHN zoo it runs 0.8494 (w16) to 0.9629 (w384) under SP,
where a raw tau curve is genuinely not comparable across widths -- there, normalizing changes
conclusions (ScaleGMN SP baseline's raw tau *rises* w16 -> w96 while its normalized tau already
falls). The SVHN doc has no BEGIN/END markers, so `--dataset svhn --stdout` is its entry point and
its block is pasted by hand.

Only the text between the `<!-- BEGIN predgen-v2-results -->` and `<!-- END ... -->` markers
is touched, so the surrounding prose (which needs judgement, not parsing) is safe. Idempotent
and incremental: run it as often as you like, including while the queue is still draining, and
it always reflects what has finished. It writes nothing if the block would not change.

Interpretation is deliberately NOT automated. The block it writes is numbers plus the
`v1 agreement` audit, and it ends with a pointer saying the analysis is still to be written.

Usage:
    python scripts/render_predgen_v2_results.py            # render whatever has finished
    python scripts/render_predgen_v2_results.py --stdout    # print, do not edit the doc
"""
import argparse
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_cnn_zoo_quality import (  # noqa: E402
    labels, read_meta, set_dataset, tau_b_ceiling,
)

#: ``--summary``/``--doc`` defaults per ``--dataset``: each study's widest re-score, i.e. its
#: Stage 3e queue. Pass ``--summary .../predgen_svhn_sizegen_v1/SUMMARY.md`` to render the
#: narrower w16-w128 SVHN queue instead; both shapes render (the spotlight widths and the
#: ``N x`` reach are derived from whatever widths were scored).
DEFAULTS = {
    "cifar10": (Path("/tmp/predgen_sizegen_v2/SUMMARY.md"),
                ROOT / "results/EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md"),
    "svhn": (Path("/tmp/predgen_svhn_sizegen_v2/SUMMARY.md"),
             ROOT / "results/EXPERIMENTS_SVHN_CNN_accuracy_prediction.md"),
}
BEGIN = "<!-- BEGIN predgen-v2-results -->"
END = "<!-- END predgen-v2-results -->"

# The 11th condition, scored by run_{cifar10,svhn}_predgen_mpsgmn_sizegen.sh rather than by
# this queue. Kept in the glance table because it is the best condition on both zoos and the
# reference point. Positional to match render()'s rows, and hardcoded to (w16, w128, w512) --
# so each list is valid only for its dataset's *v2* summary, and render() is told when to
# include it. Columns: w16 tau, w128 tau, w512 tau, mean OOD tau, w512 R2, v1 agreement.
EXTRA_ROWS = {
    "cifar10": [
        ("11", "mpsgmn, muP16, dup-equiv", 0.9062, 0.9256, 0.9123, 0.9219, 0.913,
         "exact (w16–w128)"),
        ("11", "mpsgmn, SP, dup-equiv", 0.9062, 0.8655, 0.7471, 0.8509, 0.468,
         "exact (w16–w128)"),
    ],
    "svhn": [
        ("11", "mpsgmn, muP16, dup-equiv", 0.7788, 0.8308, 0.8383, 0.8332, 0.913,
         "exact (w16–w128)"),
        ("11", "mpsgmn, SP, dup-equiv", 0.7788, 0.8558, 0.7752, 0.8383, 0.797,
         "exact (w16–w128)"),
    ],
}


def parse_summary(text):
    """-> [ {label, ood, agreement, widths: {w: (tau, r2, l1, r2rc)}} ] in file order."""
    conditions = []
    for chunk in text.split("\n### ")[1:]:
        lines = chunk.splitlines()
        cond = {"label": lines[0].strip(), "ood": None, "agreement": "?", "widths": {}}
        for line in lines[1:]:
            if m := re.match(r"- mean OOD tau \(w32-w512\): \*\*([-0-9.]+)\*\*", line):
                cond["ood"] = float(m.group(1))
            elif m := re.match(r"- (v1 agreement: .*|no v1 numbers.*|MISMATCH.*)", line):
                cond["agreement"] = m.group(1)
            elif m := re.match(
                r"\| w(\d+) \((?:IN|OUT) *\) \| ([-0-9.]+) \| ([-+0-9.]+) \| ([0-9.]+) \| ([-+0-9.]+) \|",
                line,
            ):
                w, tau, r2, l1, r2rc = m.groups()
                cond["widths"][int(w)] = (float(tau), float(r2), float(l1), float(r2rc))
        if cond["widths"]:
            conditions.append(cond)
    return conditions


def signed(v, places, plus=False):
    """Format with the doc's en-dash minus; `plus` also writes an explicit +."""
    sign = "−" if v < 0 else ("+" if plus else "")
    return f"{sign}{abs(v):.{places}f}"


def arm_of(label):
    """Which zoo arm a condition's inputs came from — 'sp' or 'mup16' — read off its label."""
    low = label.lower()
    if "mup16" in low:
        return "mup16"
    if re.search(r"\bsp\b", low):
        return "sp"
    raise ValueError(f"cannot tell the zoo arm from condition label {label!r}")


def frac_attainable(tau, width, arm):
    """``tau / tau_b_max(width, arm)`` — None if that cohort's labels aren't on disk."""
    if tau is None:
        return None
    ceiling = tau_b_ceiling(width, arm)[1]
    return None if math.isnan(ceiling) else tau / ceiling


def label_sd(width, arm):
    """Sample sd of one cohort's labels — R²'s denominator, hence its per-width reference."""
    y = labels(read_meta(width, arm, "test"))
    return y.std(ddof=1) if len(y) > 1 else float("nan")


def agreement_cell(agreement):
    if agreement.startswith("v1 agreement: OK"):
        return "exact (w16–w128)"
    if agreement.startswith("MISMATCH"):
        return f"**{agreement}**"
    # "?" = the summary carries no audit line, which is the normal case for a v1 queue:
    # v1 *is* the reference the v2 re-score is later checked against.
    return "—" if agreement == "?" else agreement


def render(conditions, finished, summary, extras=()):
    out = []
    n = len(conditions)
    if n == 0:
        out.append(f"*No condition has finished yet. Progress: `{summary}`.*")
        return "\n".join(out) + "\n"

    out.append(
        f"**{n} of 10 conditions scored"
        + ("; the queue is drained." if finished else ", queue still draining.")
        + f"** Rendered from `{summary}` by"
        " `scripts/render_predgen_v2_results.py`, which only writes the numbers — the analysis"
        " below the tables is written by hand and may lag the table."
    )
    out.append("")
    # The narrow and far widths are read off the data, not hardcoded: the v2 CIFAR-10 queue
    # reaches w512 while a v1 queue stops at w128, and both must render.
    widths = sorted({w for c in conditions for w in c["widths"]})
    if len(widths) < 3:
        sys.exit(f"need at least 3 scored widths to render a glance table, got {widths}")
    w_in, w_far = widths[0], widths[-1]
    # strictly interior, or the middle column duplicates w_far on a queue that stops at w128
    interior = widths[1:-1]
    w_mid = 128 if 128 in interior else interior[len(interior) // 2]
    out.append(f"| # | Condition | w{w_in} τ | w{w_mid} τ | w{w_far} τ | mean OOD τ |"
               f" Δτ (w{w_in}→w{w_far}) | Δ(τ/τ_b^max) | w{w_far} R² | v1 agreement |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")

    rows = []
    for i, c in enumerate(conditions, start=1):
        w = c["widths"]
        if not {w_in, w_far}.issubset(w):
            continue
        rows.append(
            (
                str(i),
                c["label"],
                w[w_in][0],
                w[w_mid][0] if w_mid in w else None,
                w[w_far][0],
                c["ood"],
                w[w_far][1],
                agreement_cell(c["agreement"]),
            )
        )
    rows.extend(extras)
    best_ood = max((r[5] for r in rows if r[5] is not None), default=None)
    for num, label, t_in, t_mid, t_far, ood, r2, agree in rows:
        # Doc convention: Delta tau = tau(w_in) - tau(w_far), so POSITIVE means tau fell.
        d = t_in - t_far
        arm = arm_of(label)
        a_in = frac_attainable(t_in, w_in, arm)
        a_far = frac_attainable(t_far, w_far, arm)
        d_norm = None if a_in is None or a_far is None else a_in - a_far
        f = lambda v, p=4: "n/a" if v is None else f"{v:.{p}f}"
        bold = ood is not None and best_ood is not None and abs(ood - best_ood) < 1e-9
        ood_s = f"**{f(ood)}**" if bold else f(ood)
        out.append(
            f"| {num} | {label} | {f(t_in)} | {f(t_mid)} | {f(t_far)} | {ood_s} | "
            f"{signed(d, 4)} | {'n/a' if d_norm is None else signed(d_norm, 4)} | "
            f"{signed(r2, 3, plus=True)} | {agree} |"
        )
    out.append("")
    arms = sorted({arm_of(c["label"]) for c in conditions})
    ceilings = [tau_b_ceiling(w, a)[1] for w in widths for a in arms]
    ceilings = [c for c in ceilings if not math.isnan(c)]
    span = (f"{min(ceilings):.4f}–{max(ceilings):.4f}" if ceilings else "unavailable")
    out.append(
        f"Δτ is the in-distribution-to-{w_far // w_in}× drop (positive = τ fell)."
        " Δ(τ/τ_b^max) is the same drop after dividing each width's τ by the largest τ_b its own"
        " labels admit, which is the width-comparable version: tied labels cap τ_b below 1 by an"
        " amount set by the label distribution, not by the metanetwork. This zoo's ceiling runs"
        f" {span} over the widths and arms below, so the closer that range sits to 1 the more"
        " the two Δ columns agree. Rows are the checkpoints re-scored, in the queue's order"
        + ("; the last row is the conv matrix-product ScaleGMN, scored by its own runner and"
           " shown for comparison" if extras else "")
        + (". Every `v1 agreement` entry must read `exact` — it is the audit that the checkpoint,"
           " data and splits are the ones v1 measured."
           if any(c["agreement"] != "?" for c in conditions)
           else ". This queue carries no `v1 agreement` audit: it *is* the v1 reference that a"
                " later re-score is checked against.")
    )
    out.append("")
    out.append("#### Per-width detail")
    out.append("")
    header = "| Condition | metric | " + " | ".join(f"w{w}" for w in widths) + " |"
    out.append(header)
    out.append("|---|---|" + "---|" * len(widths))
    for c in conditions:
        arm = arm_of(c["label"])
        # R2/L1 sit at v[1]/v[2]/v[3]; the normalized tau is derived, so it has no v index.
        for idx, metric in zip([0, None, 1, 2, 3], ["τ", "τ/τ_b^max", "R²", "L1", "R²_recal"]):
            cells = []
            for w in widths:
                v = c["widths"].get(w)
                if v is None:
                    cells.append("—")
                elif metric == "τ":
                    cells.append(f"{v[0]:.4f}")
                elif metric == "τ/τ_b^max":
                    n = frac_attainable(v[0], w, arm)
                    cells.append("—" if n is None else f"{n:.4f}")
                elif metric == "L1":
                    cells.append(f"{v[2]:.4f}")
                else:
                    cells.append(signed(v[idx], 3, plus=True))
            label = c["label"] if metric == "τ" else ""
            out.append(f"| {label} | {metric} | " + " | ".join(cells) + " |")
    out.append("")
    sds = {a: (label_sd(w_in, a), label_sd(w_far, a)) for a in arms}
    sd_txt = "; ".join(f"{a} {lo:.3f} → {hi:.3f}" for a, (lo, hi) in sds.items()
                       if not (math.isnan(lo) or math.isnan(hi)))
    out.append(
        "Only τ carries a label-tie ceiling, so only τ is normalized. R² has no ceiling below 1,"
        " but it is divided by *this width's* label variance, so it is not width-comparable"
        f" either: `sd(y)` runs w{w_in} → w{w_far} as {sd_txt or 'unavailable'}, and a width whose"
        " labels spread out is flattered for equal absolute error. L1 is normalized by nothing and"
        " is the one raw cross-width number. R²_recal is scale- and shift-invariant, so it is"
        " immune to the mean-shift that drives R² negative but still reads against a per-width"
        " variance."
    )
    out.append("")
    if not finished:
        out.append(
            f"*{10 - n} conditions still queued; re-run"
            " `python scripts/render_predgen_v2_results.py` to refresh this block (the watcher"
            " in screen `predgen_v2_render` does it every 5 min).*"
        )
        out.append("")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true", help="print the block, do not edit the doc")
    ap.add_argument("--dataset", choices=sorted(DEFAULTS), default="cifar10",
                    help="which study's zoo the tau_b ceilings are read from")
    ap.add_argument("--summary", type=Path, default=None, help="override the SUMMARY.md path")
    ap.add_argument("--doc", type=Path, default=None, help="override the results doc to edit")
    args = ap.parse_args()

    default_summary, default_doc = DEFAULTS[args.dataset]
    summary = args.summary or default_summary
    doc_path = args.doc or default_doc
    set_dataset(args.dataset)  # pins which zoo tree the ceilings are read from

    if not summary.exists():
        sys.exit(f"no summary at {summary} -- has the {args.dataset} size-gen queue started?")
    text = summary.read_text()
    # EXTRA_ROWS is hardcoded to (w16, w128, w512), so it belongs to the v2 summary only --
    # a v1 queue stops at w128 and would put the w512 numbers in the wrong columns.
    extras = EXTRA_ROWS[args.dataset] if summary == DEFAULTS[args.dataset][0] else []
    block = render(parse_summary(text), "\nFinished:" in text, summary, extras)

    if args.stdout:
        print(block, end="")
        return

    doc = doc_path.read_text()
    if BEGIN not in doc or END not in doc:
        sys.exit(f"markers {BEGIN} / {END} not found in {doc_path}")
    head, rest = doc.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{BEGIN}\n\n{block}\n{END}{tail}"
    if "\nFinished:" in text and doc_path == DEFAULTS["cifar10"][1]:
        new = flip_status_lines(new)
    if new == doc:
        print("no change")
        return
    doc_path.write_text(new)
    print(f"updated {doc_path.relative_to(ROOT)}")


def flip_status_lines(doc):
    """Once the queue is drained, retire the two 'running' phrasings outside the block.

    Exact-match substitutions, so an edit to the surrounding prose makes this a loud no-op
    rather than a silent corruption -- the numbers in the marked block are authoritative
    either way.
    """
    subs = [
        (
            "### Results — All Ten Conditions (draining since 2026-08-29 09:30)",
            "### Results — All Ten Conditions",
        ),
        (
            "| 3e | v2 re-score, 10 conditions on w16–w512 | **RUNNING since 2026-08-29 09:30** — one worker,",
            "| 3e | v2 re-score, 10 conditions on w16–w512 | **DONE — all ten scored** (queue drained; see the"
            " [results table](#results--all-ten-conditions), machine-rendered) — one worker,",
        ),
    ]
    for old, new_text in subs:
        if old in doc:
            doc = doc.replace(old, new_text, 1)
        elif new_text not in doc:
            print(f"WARNING: could not flip status text (prose changed?): {old[:60]}...")
    return doc


if __name__ == "__main__":
    main()
