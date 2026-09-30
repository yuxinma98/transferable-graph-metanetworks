"""
Evaluate trained sizegen models on widened INRs (Experiment 2).

Loads the size-gen checkpoints of the requested dataset — MNIST size-gen v3 (train w24) or
FMNIST size-gen v2 (train w32), in both cases per-width-LR SP fits and muP fits whose base
width is the training width — and evaluates on:
- the base-width original test set (in-distribution reference)
- the same test INRs widened to 2/4/6/8/10x the base width

The widened INRs compute the exact same function as the base-width originals for every
widening family (`scripts/generate_duplicated_inrs.py`), so any accuracy change is the
metanetwork failing to be invariant to the widening — never a change in the input networks'
behaviour.

Which family a model is invariant to is the point of the comparison:
- `uniform`  (Kronecker, `_dup/`): all 11 conditions have a chance; it is the family the
  edgewise-MLP GMNs' fan-in rescaling + mean aggregation + layer-wise mean readout is built for.
- `general`  (`_gen/`): the largest function-preserving family (row condition only). Only the
  FORWARD matrix-product models are invariant to it; the edgewise-MLP GMNs are not.

11 conditions per parameterization arm:
    ScaleGMN  {baseline, dup-equiv} x {forward, bidirectional}   (4)
    plain GMN {baseline, dup-equiv} x {forward, bidirectional}   (4)
    mp-GMN    dup-equiv x {forward, bidirectional}               (2)
    mp-ScaleGMN dup-equiv x forward                              (1)
The matrix-product models are dup-equiv only (the MSG constraint is defined on top of fan-in
rescaling + mean aggregation), and mp-ScaleGMN is forward-only in every experiment.

Usage:
    # every model type, one wandb run per (family, init-type, direction)
    python scripts/eval_sizegen_duplicated.py --init-type mup24 --family general \
        --model-type all --direction forward --wandb True
    # a single condition group
    python scripts/eval_sizegen_duplicated.py --init-type sp --model-type mpgmn \
        --direction bidirectional --family general
    # FMNIST: base width 32, muP base_width=32, size-gen v2 checkpoints
    python scripts/eval_sizegen_duplicated.py --dataset fmnist --init-type mup32 \
        --family general --model-type all --direction forward --wandb True
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
SCALEGMN_ROOT = PROJECT_ROOT / "src" / "scalegmn"

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

os.chdir(SCALEGMN_ROOT)
sys.path.insert(0, str(SCALEGMN_ROOT))

import torch_geometric  # noqa: E402

from src.scalegmn.models import ScaleGMN  # noqa: E402
from src.data.mnist_inr_dataset import LabeledINRDataset  # noqa: E402
from data.duplication_equiv_dataset import FanInLabeledINRDataset  # noqa: E402
from models.layerwise_mean_readout import install_layerwise_mean_readout  # noqa: E402
from models.matrix_product_scale_layer import matrix_product_context  # noqa: E402
from models.bidir_reciprocal import install_bidir_reciprocal  # noqa: E402
from src.utils.helpers import (  # noqa: E402
    mask_hidden,
    mask_input,
    overwrite_conf,
    set_seed,
)

MODEL_TYPES = ("scalegmn", "gmn", "mpgmn", "mpsgmn")

# Model types whose message function is only defined on top of fan-in rescaling + mean
# aggregation, so they have no baseline variant.
DEQ_ONLY = ("mpgmn", "mpsgmn")

# mp-ScaleGMN is forward-only in every experiment (src/models/matrix_product_scale_layer.py).
FORWARD_ONLY = ("mpsgmn",)

LABELS = {"scalegmn": "ScaleGMN", "gmn": "GMN", "mpgmn": "mp-GMN", "mpsgmn": "mp-ScaleGMN"}

# Per-dataset settings. `base_width` is the training width of the checkpoints below, which is
# also the muP base width, so the muP arm reduces exactly to SP at the base width; `tag` is
# the config generation holding that dataset's size-generalization experiment.
DATASETS = {
    "mnist": {
        "inr_dir": "mnist_inrs",
        "config_dir": "mnist_cls",
        "tag": "v3",
        "split_name": "mnist_sizegen_splits.json",
        "wandb_project": "mnist_cls",
        "run_prefix": "sizegen-v3",
        "init_types": ("sp", "mup24"),
        "base_width": 24,
        "target_widths": [48, 96, 144, 192, 240],
    },
    "fmnist": {
        "inr_dir": "fmnist_inrs",
        "config_dir": "fmnist_cls",
        "tag": "v2",
        "split_name": "fmnist_sizegen_splits_v2.json",
        "wandb_project": "fmnist_cls",
        "run_prefix": "sizegen-fmnist-v2",
        "init_types": ("sp", "mup32"),
        "base_width": 32,
        "target_widths": [64, 128, 192, 256, 320],
    },
}

# Size-gen checkpoints, keyed by dataset and then by
# (model_type, init_type, direction, variant); every one was produced by
# `train_sizegen.py --conf configs/{config_dir}/{model}_sizegen_{init}_{tag}.yml` with
# `--direction` / `--duplication-equiv` as CLI overrides, so one config serves all four
# (direction, variant) cells of a (model, init) pair.
CHECKPOINTS = {}

CHECKPOINTS["mnist"] = {
    # ScaleGMN forward
    ("scalegmn", "sp", "forward", "baseline"): "sizegen-v3-sgmn-sp-base-full_f8f8r0zp",
    ("scalegmn", "sp", "forward", "dup-equiv"): "sizegen-v3-sgmn-sp-deq-full_bngbeln2",
    ("scalegmn", "mup24", "forward", "baseline"): "sizegen-v3-sgmn-mup-base_2ao0wbnm",
    ("scalegmn", "mup24", "forward", "dup-equiv"): "sizegen-v3-sgmn-mup-deq_vgr92oh9",
    # ScaleGMN bidirectional — the `-recip` retrains; the pre-fix runs are not
    # scale-equivariant (src/models/bidir_reciprocal.py).
    ("scalegmn", "sp", "bidirectional", "baseline"):
        "sizegen-v3-recip-sgmn-sp-base-bidir-recip_7ubil4t3",
    ("scalegmn", "sp", "bidirectional", "dup-equiv"):
        "sizegen-v3-recip-sgmn-sp-deq-bidir-recip_j73y79y1",
    ("scalegmn", "mup24", "bidirectional", "baseline"):
        "sizegen-v3-recip-sgmn-mup24-base-bidir-recip_qxx0ybx2",
    ("scalegmn", "mup24", "bidirectional", "dup-equiv"):
        "sizegen-v3-recip-sgmn-mup24-deq-bidir-recip_dsxv4qxi",
    # plain GMN
    ("gmn", "sp", "forward", "baseline"): "sizegen-v3-gmn-sp-base-full_lqrhopx1",
    ("gmn", "sp", "forward", "dup-equiv"): "sizegen-v3-gmn-sp-deq-full_c9y36coc",
    ("gmn", "mup24", "forward", "baseline"): "sizegen-v3-gmn-mup-base_to0udreq",
    ("gmn", "mup24", "forward", "dup-equiv"): "sizegen-v3-gmn-mup-deq_ikmnmpj4",
    ("gmn", "sp", "bidirectional", "baseline"): "sizegen-v3-gmn-sp-base-bidir_yu8rk2nv",
    ("gmn", "sp", "bidirectional", "dup-equiv"): "sizegen-v3-gmn-sp-deq-bidir_tuiq98sc",
    ("gmn", "mup24", "bidirectional", "baseline"): "sizegen-v3-gmn-mup24-base-bidir_45215zqm",
    ("gmn", "mup24", "bidirectional", "dup-equiv"): "sizegen-v3-gmn-mup24-deq-bidir_u8oc73z3",
    # matrix-product GMN (dup-equiv only)
    ("mpgmn", "sp", "forward", "dup-equiv"): "sizegen-v3-mpgmn-sp-fw_u1ox1bd0",
    ("mpgmn", "mup24", "forward", "dup-equiv"): "sizegen-v3-mpgmn-mup24-fw_ktv38f0r",
    ("mpgmn", "sp", "bidirectional", "dup-equiv"): "sizegen-v3-mpgmn-sp-bidir_it3vpqm0",
    ("mpgmn", "mup24", "bidirectional", "dup-equiv"): "sizegen-v3-mpgmn-mup24-bidir_aid3oqcf",
    # matrix-product ScaleGMN (dup-equiv, forward only)
    ("mpsgmn", "sp", "forward", "dup-equiv"): "sizegen-v3-mpsgmn-sp_w6a7k0zd",
    ("mpsgmn", "mup24", "forward", "dup-equiv"): "sizegen-v3-mpsgmn-mup24_g451pl6g",
}

# FMNIST size-gen v2 checkpoints (train w32, muP base_width=32).
CHECKPOINTS["fmnist"] = {
    # ScaleGMN forward
    ("scalegmn", "sp", "forward", "baseline"): "sizegen-fmnist-v2-sgmn-sp-base_jbd9luuz",
    ("scalegmn", "sp", "forward", "dup-equiv"): "sizegen-fmnist-v2-sgmn-sp-deq_f33nszbu",
    ("scalegmn", "mup32", "forward", "baseline"): "sizegen-fmnist-v2-sgmn-mup32-base_4e850ho2",
    ("scalegmn", "mup32", "forward", "dup-equiv"): "sizegen-fmnist-v2-sgmn-mup32-deq_go46ekpy",
    # ScaleGMN bidirectional — the `-recip` retrains, as on MNIST.
    ("scalegmn", "sp", "bidirectional", "baseline"):
        "sizegen-fmnist-v2-recip-sgmn-sp-base-bidir-recip_wsu0u4lj",
    ("scalegmn", "sp", "bidirectional", "dup-equiv"):
        "sizegen-fmnist-v2-recip-sgmn-sp-deq-bidir-recip_t7p9n6bb",
    ("scalegmn", "mup32", "bidirectional", "baseline"):
        "sizegen-fmnist-v2-recip-sgmn-mup32-base-bidir-recip_5gyysz75",
    ("scalegmn", "mup32", "bidirectional", "dup-equiv"):
        "sizegen-fmnist-v2-recip-sgmn-mup32-deq-bidir-recip_38r10ibl",
    # plain GMN
    ("gmn", "sp", "forward", "baseline"): "sizegen-fmnist-v2-gmn-sp-base_4595gtdh",
    ("gmn", "sp", "forward", "dup-equiv"): "sizegen-fmnist-v2-gmn-sp-deq_ubsaeszh",
    ("gmn", "mup32", "forward", "baseline"): "sizegen-fmnist-v2-gmn-mup32-base_yyl5ku0p",
    ("gmn", "mup32", "forward", "dup-equiv"): "sizegen-fmnist-v2-gmn-mup32-deq_rm86w1di",
    ("gmn", "sp", "bidirectional", "baseline"): "sizegen-fmnist-v2-gmn-sp-base-bidir_hw8n6bql",
    ("gmn", "sp", "bidirectional", "dup-equiv"): "sizegen-fmnist-v2-gmn-sp-deq-bidir_tvl5gdjl",
    ("gmn", "mup32", "bidirectional", "baseline"):
        "sizegen-fmnist-v2-gmn-mup32-base-bidir_dlukne93",
    ("gmn", "mup32", "bidirectional", "dup-equiv"):
        "sizegen-fmnist-v2-gmn-mup32-deq-bidir_3saj2l9f",
    # matrix-product GMN (dup-equiv only)
    ("mpgmn", "sp", "forward", "dup-equiv"): "sizegen-fmnist-v2-mpgmn-sp_gvq4iwp6",
    ("mpgmn", "mup32", "forward", "dup-equiv"): "sizegen-fmnist-v2-mpgmn-mup32_qkvcq7qs",
    ("mpgmn", "sp", "bidirectional", "dup-equiv"): "sizegen-fmnist-v2-mpgmn-sp-bidir_nonii5lk",
    ("mpgmn", "mup32", "bidirectional", "dup-equiv"):
        "sizegen-fmnist-v2-mpgmn-mup32-bidir_fi9lrcgp",
    # matrix-product ScaleGMN (dup-equiv, forward only)
    ("mpsgmn", "sp", "forward", "dup-equiv"): "sizegen-fmnist-v2-mpsgmn-sp_odojljv7",
    ("mpsgmn", "mup32", "forward", "dup-equiv"): "sizegen-fmnist-v2-mpsgmn-mup32_a41u4mq3",
}


def conditions_for(dataset, model_types, init_type, direction):
    """[(label, model_type, variant, ckpt_name)] for the requested cells."""
    out = []
    for mt in model_types:
        if direction == "bidirectional" and mt in FORWARD_ONLY:
            print(f"  [SKIP] {LABELS[mt]} is forward-only in every experiment "
                  f"(no bidirectional checkpoint exists).")
            continue
        variants = ("dup-equiv",) if mt in DEQ_ONLY else ("baseline", "dup-equiv")
        for variant in variants:
            ckpt = CHECKPOINTS[dataset][(mt, init_type, direction, variant)]
            out.append((f"{LABELS[mt]} {variant}", mt, variant, ckpt))
    return out


def resolve_data_paths(conf: dict) -> dict:
    data = conf.get("data", {})
    for key in ("train_dataset_paths", "train_split_paths", "test_dataset_paths", "test_split_paths"):
        resolved = []
        for p in data.get(key, []):
            expanded = os.path.expandvars(p)
            if not Path(expanded).is_absolute():
                expanded = str((PROJECT_ROOT / expanded).resolve())
            resolved.append(expanded)
        data[key] = resolved
    return conf


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    total_correct = total_n = 0
    for batch in loader:
        batch = batch.to(device)
        out = net(batch)
        preds = out.argmax(-1)
        total_correct += (preds == batch.label).sum().item()
        total_n += batch.num_graphs
    return total_correct / total_n


def build_model_and_load(conf, ckpt_path, device, fanin_rescale):
    if fanin_rescale:
        conf["scalegmn_args"]["gnn_args"]["aggregator"] = "mean"

    with matrix_product_context(conf["scalegmn_args"]):
        net = ScaleGMN(conf["scalegmn_args"]).to(device)
    if fanin_rescale:
        install_layerwise_mean_readout(net, conf["scalegmn_args"])
    if install_bidir_reciprocal(net, conf["scalegmn_args"]):
        print("  Installed reciprocal backward edge features (bidirectional scale equivariance).")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net


def build_test_loader(conf_data, layout, dpath, spath, direction, equiv_on_hidden,
                      get_first_layer_mask, fanin_rescale, batch_size):
    dataset_cls = FanInLabeledINRDataset if fanin_rescale else LabeledINRDataset
    ds = dataset_cls(
        dataset=conf_data["dataset"],
        dataset_path=dpath,
        split_path=spath,
        debug=False,
        split="test",
        node_pos_embed=conf_data.get("node_pos_embed", False),
        edge_pos_embed=conf_data.get("edge_pos_embed", False),
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=get_first_layer_mask,
        image_size=tuple(conf_data.get("image_size", [28, 28])),
        layer_layout=layout,
        direction=direction,
        switch_to_canon=False,
    )
    return torch_geometric.loader.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="mnist")
    parser.add_argument("--init-type", required=True,
                        help="Parameterization arm; per dataset, one of DATASETS[ds]['init_types']")
    parser.add_argument("--model-type", choices=MODEL_TYPES + ("all",), default="all")
    parser.add_argument("--direction", choices=["forward", "bidirectional"], default="forward")
    parser.add_argument("--family", default="general",
                        help="Widening family, as named by generate_duplicated_inrs.py; "
                             "selects the w{N}_{init}_{suffix}/ data dirs")
    parser.add_argument("--wandb", type=lambda v: v.lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--base-width", type=int, default=None,
                        help="Defaults to the dataset's training width")
    parser.add_argument("--target-widths", type=int, nargs="+", default=None,
                        help="Defaults to 2/4/6/8/10x the base width")
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    assert args.init_type in ds["init_types"], \
        f"--init-type for {args.dataset} must be one of {ds['init_types']}"
    if args.base_width is None:
        args.base_width = ds["base_width"]
    if args.target_widths is None:
        args.target_widths = ds["target_widths"]

    # Loaded by path rather than by name: the generator owns the family -> dir-suffix
    # mapping (so the two scripts cannot disagree about which directory holds which
    # family), but `scripts/` must not go on sys.path here — we have already chdir'd into
    # src/scalegmn/ and `src` is bound to ScaleGMN's package.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_duplicated_inrs", PROJECT_ROOT / "scripts" / "generate_duplicated_inrs.py")
    gdi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gdi)
    assert args.family in gdi.FAMILIES, f"--family must be one of {gdi.FAMILIES}"
    suffix = gdi.FAMILY_SUFFIX[args.family]

    data_root = Path(os.environ.get("ANYDIM_DATA_ROOT", "data")) / ds["inr_dir"]
    ckpt_root = Path(os.environ.get("ANYDIM_DATA_ROOT", "data")) / "checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    set_seed(0)

    model_types = MODEL_TYPES if args.model_type == "all" else (args.model_type,)
    print(f"Dataset: {args.dataset} (base width {args.base_width}), "
          f"family: {args.family} (w{{N}}_{args.init_type}_{suffix}/), "
          f"init: {args.init_type}, direction: {args.direction}")
    conditions = conditions_for(args.dataset, model_types, args.init_type, args.direction)

    run_id = "-"
    if args.wandb:
        import wandb
        wandb.init(
            project=ds["wandb_project"],
            name=f"{ds['run_prefix']}-{suffix}-eval-{args.model_type}-{args.init_type}"
                 f"-{args.direction}",
            tags=[f"{ds['run_prefix']}-widen-eval", args.family, args.model_type,
                  args.init_type, args.direction],
            config={"dataset": args.dataset, "family": args.family,
                    "init_type": args.init_type,
                    "direction": args.direction, "base_width": args.base_width,
                    "target_widths": args.target_widths},
        )
        run_id = wandb.run.id
    # Grepped by run_mnist_cls_widen_families_eval.sh to record the run ID in its summary.
    print(f"WANDB_RUN_ID: {run_id}")

    test_widths = [args.base_width] + args.target_widths

    print()
    print(f"{'Condition':<24} | " + " | ".join(f"w{w:>4}" for w in test_widths)
          + " | max_drop")
    print("-" * (27 + 9 * len(test_widths) + 11))

    all_results = {}
    for cond_name, model_type, variant, ckpt_name in conditions:
        config_path = (PROJECT_ROOT / "configs" / ds["config_dir"]
                       / f"{model_type}_sizegen_{args.init_type}_{ds['tag']}.yml")
        with open(config_path) as f:
            conf = yaml.safe_load(f)
        conf = overwrite_conf(conf, {})
        conf = resolve_data_paths(conf)
        conf["scalegmn_args"]["direction"] = args.direction

        fanin_rescale = variant == "dup-equiv"
        if fanin_rescale:
            conf["scalegmn_args"]["gnn_args"]["msg_equiv_on_hidden"] = True

        # Use the train layout for model init
        conf["scalegmn_args"]["layer_layout"] = conf["data"]["train_layer_layouts"][0]

        ckpt_path = ckpt_root / ckpt_name / "best.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] {cond_name}: checkpoint not found at {ckpt_path}")
            continue

        net = build_model_and_load(conf, ckpt_path, device, fanin_rescale)

        equiv_on_hidden = mask_hidden(conf)
        get_first_layer_mask = mask_input(conf)
        batch_size = conf["batch_size"]
        conf_data = conf["data"]

        accs = {}
        for w in test_widths:
            # The base width is read from the ORIGINAL fits, not a widening of them
            dirname = f"w{w}_{args.init_type}" if w == args.base_width \
                else f"w{w}_{args.init_type}_{suffix}"
            dpath = str(data_root / dirname)
            spath = str(data_root / dirname / ds["split_name"])

            layout = [2, w, w, 1]
            # Scale eval batch size as bs * (base_w / w)^2 to keep memory flat (as in v3 training)
            eval_bs = max(1, int(batch_size * (args.base_width ** 2) / (w ** 2)))
            loader = build_test_loader(
                conf_data, layout, dpath, spath, args.direction,
                equiv_on_hidden, get_first_layer_mask, fanin_rescale, eval_bs
            )
            accs[w] = evaluate(net, loader, device)

        wide_accs = [accs[w] for w in args.target_widths]
        max_drop = accs[args.base_width] - min(wide_accs)
        accs_str = " | ".join(f"{accs[w]:.4f}" for w in test_widths)
        print(f"{cond_name:<24} | {accs_str} | {max_drop:+.4f}")

        all_results[cond_name] = accs

        if args.wandb:
            import wandb
            for w, acc in accs.items():
                tag = "original" if w == args.base_width else "widened"
                wandb.log({f"{cond_name}/{tag}/w{w}_acc": acc})
            wandb.log({f"{cond_name}/mean_wide_acc": sum(wide_accs) / len(wide_accs),
                       f"{cond_name}/max_drop": max_drop})

    print(f"\n\nSummary ({args.family} widening, {args.init_type}, {args.direction}):")
    print("A model invariant to this family scores identically at every width.\n")
    for cond_name, accs in all_results.items():
        base_acc = accs[args.base_width]
        wide_accs = [accs[w] for w in args.target_widths if w in accs]
        mean_wide = sum(wide_accs) / len(wide_accs)
        print(f"  {cond_name}: w{args.base_width}={base_acc:.4f}, "
              f"mean_wide={mean_wide:.4f}, max_drop={base_acc - min(wide_accs):+.4f}")

    # Markdown block, extracted verbatim by the runner into its SUMMARY.md so the recorded
    # table is never retyped from the log by hand.
    print("\nBEGIN_MD")
    print("| Condition | " + " | ".join(f"w{w}" for w in test_widths) + " | max_drop |")
    print("|---" * (len(test_widths) + 2) + "|")
    for cond_name, accs in all_results.items():
        wide_accs = [accs[w] for w in args.target_widths if w in accs]
        drop = accs[args.base_width] - min(wide_accs)
        cells = " | ".join(f"{100 * accs[w]:.1f}%" for w in test_widths)
        print(f"| {cond_name} | {cells} | {100 * drop:.1f}% |")
    print("END_MD")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
