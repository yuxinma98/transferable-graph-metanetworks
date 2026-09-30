"""
Evaluate trained w16 accuracy-prediction metanetworks on widened CNNs (Experiment 2).

Stage 2 of the CNN accuracy-prediction study — the analog of
``scripts/eval_sizegen_duplicated.py`` / Experiment 2 on the INR side.

Every widened CNN in ``$ANYDIM_DATA_ROOT/{cnn_zoo,svhn_cnn_zoo}/w{N}_zoo_{dup,gen}/``
computes *exactly the same function* as its w16 original (see
``scripts/generate_duplicated_cnns.py``), and carries the original's test accuracy as
its label.  So this measures one thing with no confounds: does the prediction move when
the same function is presented at a wider architecture?

Which widening family a model is invariant to is the point of the comparison
(``results/VERIFICATION_gmn_properties.md`` § Equivalence classes):

* ``uniform`` (Kronecker, ``w{N}_zoo_dup/``) — row **and** column condition. The family the
  dup-equiv modifications (fan-in rescaling + mean aggregation, with a
  duplication-invariant readout) are built for, so it is the per-condition reference.
* ``general`` (``w{N}_zoo_gen/``) — the row condition only, i.e. the **largest**
  function-preserving widening. Only the forward matrix-product models are invariant to
  it; the edgewise-MLP GMNs are not.

**9 conditions per dataset**, all scored from w16 checkpoints trained by
``run_{cifar10,svhn}_predgen_widen_families.sh``:

    forward:        ScaleGMN {baseline, dup-equiv}, plain GMN {baseline, dup-equiv},
                    mp-GMN dup-equiv, mp-ScaleGMN dup-equiv                        (6)
    bidirectional:  plain GMN {baseline, dup-equiv}, mp-GMN dup-equiv              (3)

The matrix-product models have no baseline variant (the MSG constraint is defined on top
of fan-in rescaling + mean aggregation); mp-ScaleGMN is forward-only in every experiment;
and the bidirectional ``symmetry: scale`` conditions are **out of scope on CNN graphs**
(the reciprocal backward edge feature overflows float32 on this zoo — see the module
docstring of ``src/models/bidir_reciprocal.py``).  Those cells are out of scope, not missing.

Two families of numbers are reported per width:

* **against the labels** — Kendall tau / R2 / L1 / R2_recal, the same metrics the
  trainer reports.  A duplication-equivariant model must report the *identical* value
  at every width.
* **against the base-width predictions** — ``tau(y_base, y_w)``, ``max |dy|`` and
  ``mean |dy|``.  This is the sharp form of the prediction: dup-equiv gives
  ``tau = 1.000`` and ``max |dy| ~ 0``.  Pairing is by index, which is valid because
  every width's split lists the same base models in the same order.

``max |dy|`` is never exactly 0 even for an exactly invariant model.  On ``uniform`` the
stored widened weights carry a ~1e-7 relative rounding error at k = 3 and 6 (``1/k_in``
is not representable in float32; k = 2, 4, 8 are exact) and float32 summation reorders,
giving a measured floor of ~1e-7 — three orders of magnitude below the drift a
non-invariant model shows.  On ``general`` the floor is set by the row condition's own
float32 residual (~5e-7 absolute on O(1) blocks, so ~1e-4 relative on an O(1e-2) kernel)
and is correspondingly larger.  This script runs in full fp32 either way: TF32 (what
training uses) would put ~1e-4 of noise on the gap.

Checkpoints are resolved by run-name prefix (``{run_prefix}-{tag}_*`` under
``$ANYDIM_DATA_ROOT/checkpoints/``), so no wandb ID is hardcoded here; everything else is
read out of the checkpoint itself (``duplication_equiv``, ``readout``).  ``--ckpt
LABEL=CKPT`` overrides the table for one-off scoring.

Usage
-----
    # one job = one (family, direction) cell; scores every in-scope condition
    python scripts/eval_sizegen_predgen_duplicated.py --family general \
        --model-type all --direction forward --wandb True
    python scripts/eval_sizegen_predgen_duplicated.py --dataset svhn --family uniform \
        --model-type all --direction bidirectional --wandb True
    # explicit checkpoints, bypassing the table
    python scripts/eval_sizegen_predgen_duplicated.py --model-type scalegmn \
        --ckpt "baseline=predgen-dup-sgmn-base_ab12cd34" --wandb True

A checkpoint may be given as a full path to a ``.pt`` file, or as a directory name
under ``$ANYDIM_DATA_ROOT/checkpoints/`` (``/best.pt`` is appended).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
SCALEGMN_ROOT = PROJECT_ROOT / "src" / "scalegmn"

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

os.chdir(SCALEGMN_ROOT)
sys.path.insert(0, str(SCALEGMN_ROOT))

import torch_geometric  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from src.scalegmn.models import ScaleGMN  # noqa: E402
from data.cnn_zoo_dataset import CNNZooDataset, FanInCNNZooDataset  # noqa: E402
from models.matrix_product_scale_layer import matrix_product_context  # noqa: E402
from models.last_layer_readout import install_last_layer_readout  # noqa: E402
from models.layerwise_mean_readout import install_layerwise_mean_readout  # noqa: E402
from models.bidir_reciprocal import install_bidir_reciprocal  # noqa: E402
from src.utils.helpers import mask_hidden, mask_input, overwrite_conf, set_seed  # noqa: E402

BASE_WIDTH = 16

# The generator owns the family -> dir-tag mapping, so the two scripts cannot disagree
# about which directory holds which family. Loaded by file path: we have already chdir'd
# into src/scalegmn/, where `scripts` is not importable as a package.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "generate_duplicated_cnns", PROJECT_ROOT / "scripts" / "generate_duplicated_cnns.py")
_gdc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gdc)
FAMILIES, FAMILY_SUFFIX = _gdc.FAMILIES, _gdc.FAMILY_SUFFIX
BASE_SUFFIX = FAMILY_SUFFIX["uniform"]       # k = 1 is family-independent
SPLIT_NAME = _gdc.split_name(BASE_SUFFIX)    # cnn_zoo_dup_splits.json

DATASETS = {
    "cifar10": {"tree": "cnn_zoo", "config_dir": "cifar10_predgen",
                "wandb_project": "cifar10_predgen", "run_prefix": "predgen-dup"},
    "svhn": {"tree": "svhn_cnn_zoo", "config_dir": "svhn_predgen",
             "wandb_project": "svhn_predgen", "run_prefix": "svhn-predgen-dup"},
}

MODEL_TYPES = ("scalegmn", "gmn", "mpgmn", "mpsgmn")
LABELS = {"scalegmn": "ScaleGMN", "gmn": "GMN", "mpgmn": "mp-GMN", "mpsgmn": "mp-ScaleGMN"}
# Which h4 of the results doc a model type belongs under.
SYMMETRY_FAMILY = {"scalegmn": "ScaleGMN", "mpsgmn": "ScaleGMN",
                   "gmn": "Plain GMN", "mpgmn": "Plain GMN"}

# The matrix-product MSG is only defined on top of fan-in rescaling + mean aggregation.
DEQ_ONLY = ("mpgmn", "mpsgmn")
# mp-ScaleGMN is forward-only in every experiment (src/models/matrix_product_scale_layer.py).
FORWARD_ONLY = ("mpsgmn",)
# Bidirectional `symmetry: scale` on CNN graphs is out of scope, not pending: the
# reciprocal backward edge feature overflows float32 on this zoo
# (src/models/bidir_reciprocal.py; the ScaleGMN paper's App. A.2 reports the same).
BIDIR_OUT_OF_SCOPE = ("scalegmn",)

# Run-name tag per (model_type, direction, variant); the checkpoint directory is
# `{run_prefix}-{tag}_{wandb_id}`, resolved by glob so no ID is hardcoded.
CKPT_TAG = {
    ("scalegmn", "forward", "baseline"): "sgmn-base",
    ("scalegmn", "forward", "dup-equiv"): "sgmn-deq",
    ("gmn", "forward", "baseline"): "gmn-base",
    ("gmn", "forward", "dup-equiv"): "gmn-deq",
    ("mpgmn", "forward", "dup-equiv"): "mpgmn-deq",
    ("mpsgmn", "forward", "dup-equiv"): "mpsgmn-deq",
    ("gmn", "bidirectional", "baseline"): "gmn-base-bidir",
    ("gmn", "bidirectional", "dup-equiv"): "gmn-deq-bidir",
    ("mpgmn", "bidirectional", "dup-equiv"): "mpgmn-deq-bidir",
}


def conditions_for(model_types, direction, run_prefix):
    """[(label, model_type, variant, ckpt_prefix)] for the requested cells."""
    out = []
    for mt in model_types:
        if direction == "bidirectional" and mt in FORWARD_ONLY:
            print(f"  [SKIP] {LABELS[mt]} is forward-only in every experiment "
                  f"(no bidirectional checkpoint exists).")
            continue
        if direction == "bidirectional" and mt in BIDIR_OUT_OF_SCOPE:
            print(f"  [SKIP] bidirectional {LABELS[mt]} is out of scope on CNN graphs "
                  f"(float32 overflow in the reciprocal backward edge feature).")
            continue
        for variant in (("dup-equiv",) if mt in DEQ_ONLY else ("baseline", "dup-equiv")):
            tag = CKPT_TAG[(mt, direction, variant)]
            out.append((f"{LABELS[mt]} {variant}", mt, variant, f"{run_prefix}-{tag}"))
    return out

# Import the metric helpers from the trainer rather than re-deriving them, so the two
# scripts cannot drift apart. train_sizegen_predgen chdir's on import, which is fine
# (same target) but must happen after sys.path is set up.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from train_sizegen_predgen import regression_metrics  # noqa: E402


def layout(width):
    return [1, width, width, width, 10]


def data_dir(dup_root, width, suffix):
    """Directory + split JSON for one width of one family.

    The base width is read from the uniform tree for every family: k = 1 is
    family-independent, so `w16_zoo_dup/` is the one base-width copy on disk.
    """
    s = BASE_SUFFIX if width == BASE_WIDTH else suffix
    d = dup_root / f"w{width}_{s}"
    return d, d / _gdc.split_name(s)


def build_loader(conf, width, fanin_rescale, dup_root, suffix, max_models=None):
    conf_data = conf["data"]
    d, split_path = data_dir(dup_root, width, suffix)
    cls = FanInCNNZooDataset if fanin_rescale else CNNZooDataset
    ds = cls(
        max_items=max_models,
        dataset=conf_data["dataset"],
        dataset_path=str(d),
        split_path=str(split_path),
        split="test",
        node_pos_embed=conf_data.get("node_pos_embed", False),
        edge_pos_embed=conf_data.get("edge_pos_embed", False),
        equiv_on_hidden=mask_hidden(conf),
        get_first_layer_mask=mask_input(conf),
        layer_layout=layout(width),
        direction=conf["scalegmn_args"]["direction"],
        activation_function=conf_data.get("activation_function", "relu"),
        max_kernel_size=tuple(conf_data.get("max_kernel_size", [3, 3])),
    )
    # keep activation memory flat: the graph has O(w^2) edges
    bs = max(1, conf["batch_size"] * BASE_WIDTH ** 2 // width ** 2)
    return torch_geometric.loader.DataLoader(
        ds, batch_size=bs, shuffle=False, num_workers=conf["num_workers"],
        pin_memory=True), len(ds), bs


@torch.no_grad()
def predict(net, loader, device):
    net.eval()
    preds, actuals = [], []
    for batch in loader:
        batch = batch.to(device)
        preds.append(torch.sigmoid(net(batch)).squeeze(-1).cpu().numpy())
        actuals.append(batch.y.float().cpu().numpy())
    return np.concatenate(preds), np.concatenate(actuals)


def resolve_ckpt(spec, ckpt_root):
    p = Path(spec)
    if p.suffix == ".pt":
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    return ckpt_root / spec / "best.pt"


def resolve_ckpt_prefix(prefix, ckpt_root):
    """`{prefix}_{wandb_id}/best.pt`, newest first, or None.

    Resolving by prefix rather than by a hardcoded wandb ID keeps the runner and this
    script in sync: whatever `--run_name {prefix}` trained is what gets scored. The
    trailing `_` makes the glob unambiguous — `…-gmn-base_*` cannot match
    `…-gmn-base-bidir_…`.
    """
    cands = [d / "best.pt" for d in sorted(ckpt_root.glob(f"{prefix}_*"))
             if (d / "best.pt").exists()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def main():
    bool_arg = lambda v: v.lower() in ("true", "1", "yes")  # noqa: E731
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="cifar10")
    parser.add_argument("--model-type", choices=MODEL_TYPES + ("all",), default="all")
    parser.add_argument("--direction", choices=["forward", "bidirectional"],
                        default="forward")
    parser.add_argument("--family", choices=FAMILIES, default="uniform",
                        help="widening family, as named by generate_duplicated_cnns.py; "
                             "selects the w{N}_zoo_{dup,gen,...}/ data dirs")
    parser.add_argument("--conf", default=None,
                        help="override the config used to build the model (only valid "
                             "with a single --model-type)")
    parser.add_argument("--ckpt", action="append", default=[], metavar="LABEL=CKPT",
                        help="repeatable; bypasses the CKPT_TAG table entirely")
    parser.add_argument("--widths", type=int, nargs="+",
                        default=[16, 32, 48, 64, 96, 128])
    parser.add_argument("--dup-root", default=None,
                        help="default $ANYDIM_DATA_ROOT/cnn_zoo (svhn: svhn_cnn_zoo)")
    parser.add_argument("--max-models", type=int, default=None,
                        help="cap the models evaluated per width (smoke tests)")
    parser.add_argument("--wandb", type=bool_arg, default=False)
    parser.add_argument("--wandb-project", default=None,
                        help="default: the dataset's project (cifar10_predgen / svhn_predgen)")
    parser.add_argument("--save-predictions", default=None,
                        help="optional .npz of per-condition, per-width predictions")
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    suffix = FAMILY_SUFFIX[args.family]
    data_root = Path(os.environ.get("ANYDIM_DATA_ROOT", "data"))
    ckpt_root = data_root / "checkpoints"
    dup_root = Path(args.dup_root) if args.dup_root else data_root / ds["tree"]

    model_types = MODEL_TYPES if args.model_type == "all" else (args.model_type,)
    print(f"dataset: {args.dataset} ({dup_root}), family: {args.family} "
          f"(w{{N}}_{suffix}/), direction: {args.direction}")
    if args.ckpt:
        # explicit LABEL=CKPT: the model type is whatever --model-type says, and the
        # config/variant flags come off the checkpoint as usual
        assert args.model_type != "all", "--ckpt requires an explicit --model-type"
        for spec in args.ckpt:
            if "=" not in spec:
                raise ValueError(f"--ckpt must be LABEL=CKPT, got {spec!r}")
        conditions = [(label, args.model_type, None, resolve_ckpt(spec, ckpt_root))
                      for label, spec in (s.split("=", 1) for s in args.ckpt)]
    else:
        conditions = [(label, mt, variant, resolve_ckpt_prefix(prefix, ckpt_root))
                      for label, mt, variant, prefix
                      in conditions_for(model_types, args.direction, ds["run_prefix"])]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # "highest" (no TF32), unlike training: TF32's 10-bit mantissa puts ~1e-4 of noise
    # on the base-vs-widened prediction gap, which is only two orders below the effect
    # this script measures. Full fp32 pushes it to ~1e-6 and costs nothing here.
    torch.set_float32_matmul_precision("highest")
    set_seed(0)

    run_id = "-"
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project or ds["wandb_project"],
                   name=f"{ds['run_prefix']}-{suffix}-eval-{args.model_type}"
                        f"-{args.direction}",
                   group=f"{ds['run_prefix']}-widen-eval",
                   tags=[f"{ds['run_prefix']}-widen-eval", args.family,
                         args.model_type, args.direction],
                   config={"dataset": args.dataset, "family": args.family,
                           "direction": args.direction, "widths": args.widths})
        run_id = wandb.run.id
    # Grepped by run_{cifar10,svhn}_predgen_widen_families.sh for its summary.
    print(f"WANDB_RUN_ID: {run_id}")

    assert not (args.conf and args.model_type == "all"), \
        "--conf overrides one model's config; pass an explicit --model-type with it"

    saved = {}
    summary = {}
    groups = {}
    ckpt_info = {}
    for label, model_type, _variant, ckpt_path in conditions:
        if ckpt_path is None or not ckpt_path.exists():
            print(f"\n[SKIP] {label}: no checkpoint ({ckpt_path})")
            continue

        conf_path = Path(args.conf) if args.conf else (
            PROJECT_ROOT / "configs" / ds["config_dir"] / f"{model_type}_dup_w16.yml")
        if not conf_path.is_absolute():
            conf_path = (PROJECT_ROOT / conf_path).resolve()
        with open(conf_path) as f:
            conf = yaml.safe_load(f)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        fanin_rescale = bool(ckpt.get("duplication_equiv", False))
        readout = ckpt.get("readout", "last_layer")

        conf = overwrite_conf(conf, {})
        if fanin_rescale:
            conf["scalegmn_args"]["gnn_args"]["aggregator"] = "mean"
            conf["scalegmn_args"]["gnn_args"]["msg_equiv_on_hidden"] = True
        conf["scalegmn_args"]["direction"] = args.direction
        conf["scalegmn_args"]["readout_range"] = "full_graph"
        conf["scalegmn_args"]["layer_layout"] = layout(BASE_WIDTH)

        # no-op unless gnn_args.message_fn_type is matrix_product_conv
        with matrix_product_context(conf["scalegmn_args"]):
            net = ScaleGMN(conf["scalegmn_args"]).to(device)
        if readout == "last_layer":
            install_last_layer_readout(net, conf["scalegmn_args"])
        else:
            install_layerwise_mean_readout(net, conf["scalegmn_args"])
        # part of the definition of a bidirectional scale model; a no-op for the forward
        # Stage 2 checkpoints (see src/models/bidir_reciprocal.py)
        install_bidir_reciprocal(net, conf["scalegmn_args"])
        net = net.to(device)
        net.load_state_dict(ckpt["model_state_dict"])

        print(f"\n=== {label} ===")
        print(f"  {ckpt_path}")
        print(f"  {conf_path.name}, epoch {ckpt.get('epoch', '?')}, val_tau "
              f"{ckpt.get('val_tau', float('nan')):.4f}, "
              f"dup_equiv={fanin_rescale}, readout={readout}")
        print(f"  {'width':>6}  {'n':>5} {'bs':>4} | {'tau':>7} {'R2':>7} {'L1':>7} "
              f"{'R2rc':>7} | {'tau_base':>8} {'max|dy|':>9} {'mean|dy|':>9}")

        base_pred = None
        rows = {}
        for w in args.widths:
            loader, n, bs = build_loader(conf, w, fanin_rescale, dup_root, suffix,
                                         args.max_models)
            pred, actual = predict(net, loader, device)
            res = regression_metrics(actual, pred, loss=float(((pred - actual) ** 2).mean()))
            if w == BASE_WIDTH:
                base_pred = pred
                tau_b, max_d, mean_d = 1.0, 0.0, 0.0
            else:
                m = min(len(base_pred), len(pred))
                tau_b = float(kendalltau(base_pred[:m], pred[:m]).correlation)
                d = np.abs(base_pred[:m] - pred[:m])
                max_d, mean_d = float(d.max()), float(d.mean())
            rows[w] = dict(res, tau_base=tau_b, max_dpred=max_d, mean_dpred=mean_d, n=n)
            print(f"  w{w:<5d} {n:>5d} {bs:>4d} | {res['tau']:7.4f} {res['rsq']:+7.3f} "
                  f"{res['l1']:7.4f} {res['rsq_recal']:+7.3f} | {tau_b:8.4f} "
                  f"{max_d:9.2e} {mean_d:9.2e}", flush=True)
            if args.save_predictions:
                saved[f"{label}_w{w}_pred"] = pred
                saved[f"{label}_w{w}_actual"] = actual
            if args.wandb:
                import wandb
                for key, val in rows[w].items():
                    wandb.summary[f"{label}/w{w}_{key}"] = val

        summary[label] = rows
        groups[label] = SYMMETRY_FAMILY[model_type]
        ckpt_info[label] = (ckpt_path.parent.name, ckpt.get("val_tau", float("nan")),
                            ckpt.get("epoch", "?"))
        dup = [w for w in args.widths if w != BASE_WIDTH]
        if dup:
            print(f"  base tau {rows[BASE_WIDTH]['tau']:.4f} -> widened "
                  f"[{min(rows[w]['tau'] for w in dup):.4f}, "
                  f"{max(rows[w]['tau'] for w in dup):.4f}]   "
                  f"max over widths of max|dy| = "
                  f"{max(rows[w]['max_dpred'] for w in dup):.2e}")

    print(f"\n\nSummary — {args.family} widening, {args.direction}. "
          "A model invariant to this family reports the identical tau at every width.")
    print("\ntau against the labels, per width")
    header = f"{'condition':<24} | " + " | ".join(f"w{w:<5d}" for w in args.widths)
    print(header)
    print("-" * len(header))
    for label, rows in summary.items():
        print(f"{label:<24} | " + " | ".join(f"{rows[w]['tau']:.4f}" for w in args.widths))
    print("\nmax |dy| against the base-width prediction")
    for label, rows in summary.items():
        print(f"{label:<24} | " + " | ".join(f"{rows[w]['max_dpred']:.1e}"
                                             for w in args.widths))

    # Markdown, extracted verbatim by the runner into its SUMMARY.md so the recorded
    # tables are never retyped from the log by hand. One h4 per symmetry family, matching
    # the results doc's own sectioning; the `, fw` / `, bd` suffix is what lets the two
    # direction jobs be pasted into one table.
    dup = [w for w in args.widths if w != BASE_WIDTH]
    dir_tag = "fw" if args.direction == "forward" else "bd"
    print("\nBEGIN_MD")
    for group in ("ScaleGMN", "Plain GMN"):
        labels = [lab for lab in summary if groups[lab] == group]
        if not labels:
            continue
        print(f"\n#### {group} — {args.family} widening, {args.direction}\n")
        print("Kendall $\\tau$ against the labels:\n")
        print("| Condition | " + " | ".join(f"w{w}" for w in args.widths) + " | max_drop |")
        print("|---" * (len(args.widths) + 2) + "|")
        for lab in labels:
            rows = summary[lab]
            drop = rows[BASE_WIDTH]["tau"] - min(rows[w]["tau"] for w in dup) if dup else 0.0
            cells = " | ".join(f"{rows[w]['tau']:.4f}" for w in args.widths)
            print(f"| {lab}, {dir_tag} | {cells} | {drop:+.4f} |")
        print("\n$R^2$ / L1 / $R^2_{recal}$, w"
              f"{BASE_WIDTH} $\\to$ w{args.widths[-1]}:\n")
        print("| Condition | $R^2$ | L1 | $R^2_{recal}$ |")
        print("|---|---|---|---|")
        for lab in labels:
            rows, last = summary[lab], args.widths[-1]
            print(f"| {lab}, {dir_tag} | "
                  f"{rows[BASE_WIDTH]['rsq']:+.3f} $\\to$ {rows[last]['rsq']:+.3f} | "
                  f"{rows[BASE_WIDTH]['l1']:.4f} $\\to$ {rows[last]['l1']:.4f} | "
                  f"{rows[BASE_WIDTH]['rsq_recal']:+.3f} $\\to$ "
                  f"{rows[last]['rsq_recal']:+.3f} |")
        if dup:
            print("\nAgreement with the w16 predictions:\n")
            print("| Condition | Metric | " + " | ".join(f"w{w}" for w in dup) + " |")
            print("|---" * (len(dup) + 2) + "|")
            for lab in labels:
                rows = summary[lab]
                for key, name in (("tau_base", "`τ(y₁₆, y_w)`"),
                                  ("max_dpred", "`max \\|Δy\\|`"),
                                  ("mean_dpred", "`mean \\|Δy\\|`")):
                    fmt = "{:.4f}" if key == "tau_base" else "{:.1e}"
                    cells = " | ".join(fmt.format(rows[w][key]) for w in dup)
                    print(f"| {lab}, {dir_tag} | {name} | {cells} |")
    if ckpt_info:
        print(f"\nCheckpoints ({args.direction}), eval run `{run_id}`:\n")
        print("| Condition | Checkpoint | Best val $\\tau$ (w16) |")
        print("|---|---|---|")
        for lab, (name, val_tau, epoch) in ckpt_info.items():
            print(f"| {lab}, {dir_tag} | `{name}` | {val_tau:.4f} (ep {epoch}) |")
    print("END_MD")

    if args.save_predictions:
        np.savez(args.save_predictions, **saved)
        print(f"\npredictions -> {args.save_predictions}")
    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
