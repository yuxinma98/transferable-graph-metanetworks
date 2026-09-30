"""
Write the accuracy-prediction size-generalization splits from a generated CNN zoo.

Stage 3a of the study, run after ``scripts/generate_cnn_zoo.py``.  Emits one
``{dataset}_predgen_splits.json`` per width directory, in the ``{"path", "score"}`` layout
:class:`src.data.cnn_zoo_dataset.CNNZooDataset` reads::

    {"train":  {"path": [...], "score": [...], "score_ce": [...], ...},
     "val":    {...},
     "test":   {...},
     "paired": {...}}

``score`` is the CNN's **test accuracy** (the headline target).  ``score_ce`` (test
cross-entropy), ``score_train_acc``, ``hp`` and ``spec_norms`` ride along so the
diagnostics of Stage 3c need no second pass over the zoo; ``CNNZooDataset`` ignores them.

Structure, mirroring the INR side's ``generate_sizegen_splits.py``:

* the **train width** (w16) is the only width with ``train`` / ``val`` entries;
* **every** width has a ``test`` set of ~1k models drawn from a *width-specific* seed, so
  no hyperparameter draw is shared between two widths' test sets — the analog of the INR
  experiments' non-overlapping images.  This is asserted, not assumed;
* every width also carries the ~200-draw ``paired`` set (one shared seed, so the same
  draws exist at every width).  It is a separate key and never merges into ``test``, so
  it cannot leak into a headline number.

Because the roles and their seeds come from ``generate_cnn_zoo.role_seed``, the split and
the models on disk cannot describe different draws.  Every listed path is checked to
exist and every record is cross-checked against the HP draw its seed implies.

Usage
-----
    python scripts/generate_predgen_splits.py --arm sp    --widths 16 32 48 64 96 128
    python scripts/generate_predgen_splits.py --arm mup16 --widths 16 32 48 64 96 128
    python scripts/generate_predgen_splits.py --arm sp --widths 16 --dry-run
    python scripts/generate_predgen_splits.py --dataset svhn --arm sp --widths 16 32

Pass **every** width of the study (the default), not just newly generated ones: each
width's JSON is a deterministic function of that width's metadata, so already-written
files are reproduced byte-for-byte, but ``assert_test_disjoint`` only compares the widths
of the current invocation.  Running it on a subset would leave a new width's test draws
unchecked against the existing ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from generate_cnn_zoo import (  # noqa: E402
    BASE_WIDTH, DATASETS, ROLES, WIDTHS, read_meta, role_draws, role_seed,
    roles_for_width, width_dir,
)

#: the fields of a metadata record that become split columns, and their split keys
SCORE_KEYS = {"test_acc": "score", "test_ce": "score_ce",
              "train_acc": "score_train_acc"}


def hp_key(hp: dict) -> tuple:
    """A hashable identity for one hyperparameter draw.

    Rounded to 12 significant digits so a float32 round-trip through JSON cannot make
    two records of the *same* draw look disjoint — the assertion below has to be
    sensitive to genuinely shared draws and blind to serialization noise.
    """
    return tuple(round(float(hp[k]), 12) for k in sorted(hp))


def load_role(width: int, arm: str, role: str, dataset: str = "cifar10") -> list[dict]:
    """Records of one ``(width, arm, role)``, index-ordered, existence-checked.

    Also verifies each record against ``sample_hps(role_seed(role, width))``: the file on
    disk must be the model the split *says* it is.
    """
    wdir = width_dir(width, arm, dataset)
    recs = read_meta(wdir / f"meta_{role}.jsonl")
    if not recs:
        return []
    draws = role_draws(role, width)
    out = []
    for idx in sorted(recs):
        rec = recs[idx]
        assert hp_key(rec["hp"]) == hp_key(draws[idx].as_dict()), (
            f"w{width} {arm} {role}[{idx}]: recorded HPs do not match the draw implied "
            f"by seed {role_seed(role, width)} — the zoo and the split design disagree")
        assert (wdir / rec["path"]).exists(), f"missing {wdir / rec['path']}"
        out.append(rec)
    return out


def to_columns(recs: list[dict]) -> dict:
    """One role's records as the column-oriented split dict."""
    cols = {"path": [r["path"] for r in recs]}
    for src, dst in SCORE_KEYS.items():
        cols[dst] = [float(r[src]) for r in recs]
    cols["hp"] = [r["hp"] for r in recs]
    cols["spec_norms"] = [r["spec_norms"] for r in recs]
    cols["idx"] = [int(r["idx"]) for r in recs]
    return cols


def assert_test_disjoint(by_width: dict[int, dict[str, list[dict]]]) -> None:
    """No HP draw may appear in two widths' test sets, or in test and train/val."""
    seen: dict[tuple, str] = {}
    for role in ("train", "val"):
        for width, roles in by_width.items():
            for rec in roles.get(role, []):
                seen[hp_key(rec["hp"])] = f"w{width}/{role}"
    for width, roles in sorted(by_width.items()):
        for rec in roles.get("test", []):
            k = hp_key(rec["hp"])
            owner = f"w{width}/test"
            if k in seen:
                raise AssertionError(
                    f"hyperparameter draw shared by {seen[k]} and {owner}: {rec['hp']}")
            seen[k] = owner
    print(f"  disjointness: OK ({len(seen)} distinct draws across train/val/test)")


def summarize(width, arm, roles):
    for role in ROLES:
        recs = roles.get(role, [])
        if not recs:
            continue
        acc = np.array([r["test_acc"] for r in recs])
        q = np.quantile(acc, [0.0, 0.25, 0.5, 0.75, 1.0])
        print(f"  {role:<7} n={len(recs):<5d} test_acc min/q25/med/q75/max "
              + "  ".join(f"{v:.3f}" for v in q))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=["sp", "mup16"], required=True)
    p.add_argument("--dataset", choices=list(DATASETS), default="cifar10",
                   help="must match the generate_cnn_zoo.py run that wrote the zoo")
    p.add_argument("--widths", type=int, nargs="+", default=list(WIDTHS))
    p.add_argument("--split-name", default=None,
                   help="default: {dataset}_predgen_splits.json")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    split_name = args.split_name or f"{args.dataset}_predgen_splits.json"

    by_width: dict[int, dict[str, list[dict]]] = {}
    for width in args.widths:
        print(f"\n=== w{width} ({args.arm}) ===")
        roles = {}
        for role in roles_for_width(width):
            recs = load_role(width, args.arm, role, args.dataset)
            if not recs:
                print(f"  {role:<7} MISSING — generate it first "
                      f"(generate_cnn_zoo.py --dataset {args.dataset} "
                      f"--arm {args.arm} --widths {width})")
                continue
            expected = ROLES[role]["n"]
            if len(recs) != expected:
                print(f"  {role:<7} INCOMPLETE: {len(recs)}/{expected}")
            roles[role] = recs
        summarize(width, args.arm, roles)
        by_width[width] = roles

    if not any(r.get("train") for r in by_width.values()):
        print(f"\nNote: no width in this run has a `train` role. The train width is "
              f"w{BASE_WIDTH}; OOD widths legitimately carry test/paired only.")

    print("\n=== cross-width checks ===")
    assert_test_disjoint(by_width)

    for width, roles in sorted(by_width.items()):
        if not roles:
            continue
        out = {role: to_columns(recs) for role, recs in roles.items()}
        path = width_dir(width, args.arm, args.dataset) / split_name
        counts = "  ".join(f"{r}={len(c['path'])}" for r, c in out.items())
        if args.dry_run:
            print(f"  would write {path}  ({counts})")
        else:
            with open(path, "w") as f:
                json.dump(out, f)
            print(f"  wrote {path}  ({counts})")


if __name__ == "__main__":
    main()
