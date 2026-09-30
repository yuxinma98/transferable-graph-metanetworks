"""
Train the multi-width CNN zoo for the accuracy-prediction size-generalization study.

Stage 3a of the study.  Architecture and task are fixed to the upstream small CNN zoo's
(3 stride-2 3x3 convs, GAP, dense->10, grayscale 32x32 images); **channels** are the
width axis.  ``--dataset`` selects which images (``cifar10``, the default, or ``svhn``)
and with it the output tree, so the two studies' zoos never share a directory.  What
varies between models is a hyperparameter draw
(:func:`src.data.train_zoo_cnns.sample_hps`) — base lr, init multiplier, weight decay,
dropout, train-set fraction, epochs — and that draw is what creates the accuracy spread
the metanetwork has to predict.

Two parameterization arms, mirroring the INR side's SP vs muP24:

* ``--arm sp``    -> ``w{N}_sp/``     the sampled HPs instantiated as-is at every width.
* ``--arm mup16`` -> ``w{N}_mup16/``  the *same* base HPs with muP width scaling on top
  (``base_width = 16`` = the train width).  Because base width = train width, w16 muP16
  is **bit-identical** to w16 SP (every muP factor is exactly 1.0; verified by
  ``scripts/check_mup_cnn.py --only identity``), so the arms differ only OOD.

Roles (which models a width gets) follow the split design in
``scripts/generate_predgen_splits.py`` and are derived from one shared table, so the two
scripts cannot disagree about which draw is which:

* ``train`` / ``val`` — the train width only.  These are the metanetwork's training data.
* ``test``  — every width, from a **width-specific seed**, so no HP draw is shared
  between two widths' test sets (the analog of the INR experiments' disjoint images).
* ``paired`` — every width, from **one shared seed**, so the same ~200 draws exist at
  every width.  Never enters the headline test sets; it is what the label-shift
  (``tau_HP(16, w)``) and data-closeness (``W_1(mu_16, mu_w)``) diagnostics need.

Output layout, one directory per (width, arm) (``svhn_cnn_zoo/`` for ``--dataset svhn``)::

    $ANYDIM_DATA_ROOT/cnn_zoo/w{N}_{arm}/
        models/{role}/{idx:06d}.pth      layers.{i}.{weight,bias} state dicts
        meta_{role}.jsonl                one JSON record per model (see below)

Each metadata record carries ``idx``, ``path`` (relative to the width dir), the HP draw,
``test_acc`` (the headline target), ``test_ce``, ``train_acc``, and ``spec_norms`` (the
per-layer normalized spectral norms ``sqrt(n_{l-1}/n_l) * ||W^(l)||_2``, so the
``verify_norm_stats`` analog is free).  The file is JSONL and appended per cohort, which
is also what makes the script resumable: a cohort whose ``.pth`` files all exist and
whose records are all present is skipped.

Usage
-----
    # the train width, both roles that only it gets, SP arm
    python scripts/generate_cnn_zoo.py --arm sp --widths 16

    # OOD test + paired sets
    python scripts/generate_cnn_zoo.py --arm sp    --widths 32 48 64 96 128
    python scripts/generate_cnn_zoo.py --arm mup16 --widths 32 48 64 96 128

    # the v2 extension (32x reach); driven by scripts/run_cifar10_zoo_gen_v2.sh
    python scripts/generate_cnn_zoo.py --arm sp    --widths 192 256 384 512

    # the SVHN-GS replication; driven by scripts/run_svhn_zoo_gen.sh
    python scripts/generate_cnn_zoo.py --dataset svhn --arm sp --widths 16 32 48 64 96 128

    # w16 muP16: hard-links the SP arm (bit-identical by the base-width identity)
    python scripts/generate_cnn_zoo.py --arm mup16 --widths 16

    python scripts/generate_cnn_zoo.py --arm sp --widths 16 --dry-run   # plan only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.train_zoo_cnns import (  # noqa: E402
    GRAY_LOADERS, BatchedZooCNNTrainer, PerModelSampler, cohort_size, sample_hps,
    step_budget,
)

BASE_WIDTH = 16

#: default widths of the study (train width first).  w192-w512 were added in the v2
#: extension, which pushes the reach from 8x to 32x the train width; the ladder is
#: {3,4} * 2^k from w48 up (alternating x1.5 / x1.33), the same grid the INR size-gen
#: experiments use, so the log spacing does not thin out at the top.
WIDTHS = (16, 32, 48, 64, 96, 128, 192, 256, 384, 512)

#: CIFAR-10 (python pickle format) may live outside $ANYDIM_DATA_ROOT; point
#: $CIFAR10_ROOT at its parent if so.  ``None`` falls back to $ANYDIM_DATA_ROOT/CIFAR10.
DEFAULT_CIFAR_ROOT = os.environ.get("CIFAR10_ROOT") or None

#: ``dataset -> (zoo tree under $ANYDIM_DATA_ROOT, default image root)``.  SVHN's tree is
#: separate from CIFAR-10's so the two studies' width dirs cannot collide; ``None`` means
#: "under $ANYDIM_DATA_ROOT", resolved in :func:`image_root` once the env var is read.
DATASETS = {
    "cifar10": dict(tree="cnn_zoo", image_root=DEFAULT_CIFAR_ROOT),
    "svhn": dict(tree="svhn_cnn_zoo", image_root=None),
}

#: ``role -> (seed base, n models, per-width seed?)``.  ``per_width`` seeds add the
#: width so that no draw is shared between two widths; the shared-seed roles (train,
#: val, paired) reproduce the identical draws everywhere they are generated.
ROLES = {
    "train":  dict(seed=1000, n=8000, per_width=False, widths="train-only"),
    "val":    dict(seed=2000, n=1000, per_width=False, widths="train-only"),
    "test":   dict(seed=3000, n=1000, per_width=True,  widths="all"),
    "paired": dict(seed=9000, n=200,  per_width=False, widths="all"),
}


def roles_for_width(width: int, base_width: int = BASE_WIDTH) -> list[str]:
    """Which roles this width gets, in generation order (cheapest last)."""
    return [r for r, spec in ROLES.items()
            if spec["widths"] == "all" or width == base_width]


def role_seed(role: str, width: int) -> int:
    """The ``sample_hps`` seed for ``(role, width)`` — the single source of truth.

    ``generate_predgen_splits.py`` imports this, so a change here cannot desynchronize
    the split from the models on disk.
    """
    spec = ROLES[role]
    return spec["seed"] + (width if spec["per_width"] else 0)


def role_draws(role: str, width: int):
    """The HP draws for ``(role, width)``."""
    return sample_hps(ROLES[role]["n"], role_seed(role, width))


def data_root() -> Path:
    root = os.environ.get("ANYDIM_DATA_ROOT")
    if root is None:
        raise EnvironmentError("ANYDIM_DATA_ROOT is not set — run `source .env` first.")
    return Path(root)


def width_dir(width: int, arm: str, dataset: str = "cifar10") -> Path:
    return data_root() / DATASETS[dataset]["tree"] / f"w{width}_{arm}"


def image_root(dataset: str, override: str | None = None) -> Path:
    """Where the raw images live.  ``None`` in :data:`DATASETS` means under the data root."""
    if override is not None:
        return Path(override)
    configured = DATASETS[dataset]["image_root"]
    return Path(configured) if configured is not None else data_root() / dataset.upper()


def read_meta(path: Path) -> dict[int, dict]:
    """Existing records of one role's JSONL, keyed by ``idx`` (last write wins)."""
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["idx"]] = rec
    return out


# ---------------------------------------------------------------------------
# One cohort
# ---------------------------------------------------------------------------

def train_cohort(hps, width, arm, seed, data, batch_size, device, eval_train_items=2000):
    """Train one ``vmap`` cohort to completion and return ``(state_dicts, records)``.

    Models in a cohort have different step budgets (``epochs * train_frac``), so the
    loop runs to the longest budget and passes an ``alive`` mask to
    :meth:`BatchedZooCNNTrainer.step`; a finished model's parameters stop moving while
    its cohort-mates continue.
    """
    train_x, train_y, test_x, test_y = data
    base_width = BASE_WIDTH if arm != "sp" else None
    trainer = BatchedZooCNNTrainer(hps, width=width, base_width=base_width, seed=seed,
                                   device=device, batch_size=batch_size)
    sampler = PerModelSampler(hps, train_x.shape[0], batch_size, seed, device)

    budgets = torch.tensor([step_budget(h, train_x.shape[0], batch_size) for h in hps],
                           device=device)
    total = int(budgets.max().item())
    for t in range(total):
        alive = (budgets > t).float()
        idx = sampler.batch_indices()
        trainer.step(train_x[idx], train_y[idx], alive=alive)

    test_acc, test_ce = trainer.evaluate(test_x, test_y)
    train_acc = trainer.evaluate_per_model(
        train_x, train_y, sampler.train_eval_indices(eval_train_items))
    spec = trainer.spectral_norm_components()

    sds = trainer.export_state_dicts()
    records = [
        dict(hp=h.as_dict(), steps=int(budgets[i].item()),
             test_acc=float(test_acc[i]), test_ce=float(test_ce[i]),
             train_acc=float(train_acc[i]), spec_norms=[float(v) for v in spec[i]])
        for i, h in enumerate(hps)
    ]
    return sds, records


def generate_role(width, arm, role, data, batch_size, cohort_cap, device, dry_run,
                  force=False, limit=None, dataset="cifar10"):
    """Generate (or resume) all models of one ``(width, arm, role)``.

    ``limit`` truncates the role to its first ``limit`` draws.  Because the draws are a
    deterministic function of ``(role, width)``, a truncated run is a strict prefix of
    the full one, so raising the limit later resumes rather than restarts.
    """
    hps = role_draws(role, width)
    if limit is not None:
        hps = hps[:limit]
    wdir = width_dir(width, arm, dataset)
    mdir = wdir / "models" / role
    meta_path = wdir / f"meta_{role}.jsonl"
    have = {} if force else read_meta(meta_path)

    cohort = cohort_size(width, batch_size, cap=cohort_cap)
    n = len(hps)
    todo = [i for i in range(n)
            if i not in have or not (mdir / f"{i:06d}.pth").exists()]
    n_cohorts = math.ceil(len(todo) / cohort)
    print(f"  {role:<7} n={n:<5d} cohort={cohort:<4d} seed={role_seed(role, width):<6d} "
          f"done={n - len(todo):<5d} todo={len(todo):<5d} ({n_cohorts} cohorts)")
    if dry_run or not todo:
        return

    mdir.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "a") as meta_f:
        for c in range(n_cohorts):
            chunk = todo[c * cohort:(c + 1) * cohort]
            t0 = time.time()
            # the cohort seed is derived from the first draw index, so re-running with a
            # different cohort size still reproduces per-cohort inits deterministically
            sds, records = train_cohort([hps[i] for i in chunk], width, arm,
                                        seed=role_seed(role, width) + chunk[0],
                                        data=data, batch_size=batch_size, device=device)
            for i, sd, rec in zip(chunk, sds, records):
                torch.save(sd, mdir / f"{i:06d}.pth")
                rec["idx"] = i
                rec["path"] = f"models/{role}/{i:06d}.pth"
                meta_f.write(json.dumps(rec) + "\n")
            meta_f.flush()
            accs = [r["test_acc"] for r in records]
            print(f"    cohort {c + 1}/{n_cohorts}  {len(chunk):>4d} models  "
                  f"{time.time() - t0:7.1f}s  test_acc "
                  f"[{min(accs):.3f}, {max(accs):.3f}] mean {sum(accs) / len(accs):.3f}",
                  flush=True)


def link_base_width_mup(arm, dry_run, dataset="cifar10"):
    """Populate ``w16_{arm}/`` by hard-linking ``w16_sp/``.

    At ``width == base_width`` every muP factor is exactly 1.0 and the readout init
    multiplier is ``wm**-0.5 == 1``, so the two arms are bit-identical — the base-width
    identity that ``scripts/check_mup_cnn.py --only identity`` asserts (it passes at
    exactly 0.0, dropout masks included).  Training them twice would burn the study's
    largest single block of compute to reproduce files we already have, so they are
    hard-linked instead and the identity check is the justification.
    """
    src, dst = width_dir(BASE_WIDTH, "sp", dataset), width_dir(BASE_WIDTH, arm, dataset)
    if not src.exists():
        raise SystemExit(f"{src} does not exist — generate the SP arm first.")
    print(f"  w{BASE_WIDTH} {arm}: hard-linking from {src.name} "
          f"(base-width identity; see check_mup_cnn.py --only identity)")
    n = 0
    for role in roles_for_width(BASE_WIDTH):
        s_meta = src / f"meta_{role}.jsonl"
        if not s_meta.exists():
            print(f"    {role:<7} MISSING in the SP arm — skipped")
            continue
        if dry_run:
            print(f"    {role:<7} {len(read_meta(s_meta))} records would be linked")
            continue
        (dst / "models" / role).mkdir(parents=True, exist_ok=True)
        for rec in read_meta(s_meta).values():
            d = dst / rec["path"]
            if not d.exists():
                os.link(src / rec["path"], d)
                n += 1
        (dst / f"meta_{role}.jsonl").write_text(s_meta.read_text())
    if not dry_run:
        print(f"    {n} new hard links")


def main():
    bool_arg = lambda v: v.lower() in ("true", "1", "yes")  # noqa: E731
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=["sp", "mup16"], required=True)
    p.add_argument("--dataset", choices=list(DATASETS), default="cifar10",
                   help="which images the zoo CNNs are fitted to; also selects the "
                        "output tree (cifar10 -> cnn_zoo/, svhn -> svhn_cnn_zoo/)")
    p.add_argument("--widths", type=int, nargs="+", default=list(WIDTHS))
    p.add_argument("--roles", nargs="+", default=None, choices=list(ROLES),
                   help="default: every role this width gets")
    p.add_argument("--image-root", default=None,
                   help="parent of the raw image files; default per --dataset "
                        f"(cifar10: {DEFAULT_CIFAR_ROOT or '$ANYDIM_DATA_ROOT/CIFAR10'}, "
                        "svhn: $ANYDIM_DATA_ROOT/SVHN)")
    p.add_argument("--cifar-root", default=None,
                   help="deprecated alias for --image-root")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--cohort-cap", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true", help="print the plan, train nothing")
    p.add_argument("--force", action="store_true", help="retrain models already on disk")
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N draws of each role (smoke tests; a prefix, "
                        "so raising it later resumes)")
    p.add_argument("--no-link", type=bool_arg, default=False,
                   help="actually train w16 muP16 instead of hard-linking the SP arm")
    args = p.parse_args()

    root = image_root(args.dataset, args.image_root or args.cifar_root)
    print(f"dataset={args.dataset}  arm={args.arm}  widths={args.widths}  "
          f"device={args.device}\nzoo tree={width_dir(BASE_WIDTH, args.arm, args.dataset).parent}"
          f"  images={root}")

    link_only = [w for w in args.widths
                 if args.arm != "sp" and w == BASE_WIDTH and not args.no_link]
    train_widths = [w for w in args.widths if w not in link_only]

    data = None
    if train_widths and not args.dry_run:
        loader, label = GRAY_LOADERS[args.dataset]
        data = loader(root, device=args.device)
        print(f"{label} grayscale: train {tuple(data[0].shape)}  "
              f"test {tuple(data[2].shape)}")

    for width in args.widths:
        print(f"\n=== w{width} ({args.arm}) ===")
        if width in link_only:
            link_base_width_mup(args.arm, args.dry_run, args.dataset)
            continue
        for role in (args.roles or roles_for_width(width)):
            if role not in roles_for_width(width):
                print(f"  {role:<7} not a role of w{width} — skipped")
                continue
            generate_role(width, args.arm, role, data, args.batch_size,
                          args.cohort_cap, args.device, args.dry_run, args.force,
                          args.limit, args.dataset)

    print("\nDone. Next: python scripts/generate_predgen_splits.py "
          f"--dataset {args.dataset} --arm {args.arm} "
          f"--widths {' '.join(str(w) for w in args.widths)}")


if __name__ == "__main__":
    main()
