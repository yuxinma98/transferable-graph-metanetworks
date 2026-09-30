"""
Evaluate trained sizegen regression models on duplicated (widened) INRs.

Loads checkpoints from the 4 regression conditions and evaluates on:
- w16 original test set (in-distribution baseline)
- w32_dup, w48_dup, w64_dup, w80_dup, w96_dup (duplicated from w16 test)

Since the duplicated INRs compute the exact same function as the w16 originals,
a duplication-equivariant model should achieve identical MSE/PSNR on all widths.

Usage:
    python scripts/eval_sizegen_regression_duplicated.py --init-type sp
    python scripts/eval_sizegen_regression_duplicated.py --init-type mup
"""

import argparse
import math
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
from data.image_regression_dataset import ImageRegressionINRDataset  # noqa: E402
from models.layerwise_mean_readout import install_layerwise_mean_readout  # noqa: E402
from models.bidir_reciprocal import install_bidir_reciprocal  # noqa: E402
from src.utils.helpers import (  # noqa: E402
    mask_hidden,
    mask_input,
    overwrite_conf,
    set_seed,
)


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


def mse_to_psnr(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return -10 * math.log10(mse)


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    total_mse = 0.0
    total_n = 0
    for batch in loader:
        batch = batch.to(device)
        out = net(batch)
        B = batch.num_graphs
        target = batch.label.view(B, -1)
        mse = ((out - target) ** 2).mean(dim=-1).sum().item()
        total_mse += mse
        total_n += B
    avg_mse = total_mse / total_n
    return {"avg_mse": avg_mse, "avg_psnr": mse_to_psnr(avg_mse)}


def build_model_and_load(conf, ckpt_path, device, fanin_rescale):
    if fanin_rescale:
        conf["scalegmn_args"]["gnn_args"]["aggregator"] = "mean"

    net = ScaleGMN(conf["scalegmn_args"]).to(device)
    if fanin_rescale:
        install_layerwise_mean_readout(net, conf["scalegmn_args"])
    if install_bidir_reciprocal(net, conf["scalegmn_args"]):
        print("Installed reciprocal backward edge features (bidirectional scale equivariance).")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net


def build_test_loader(conf_data, layout, dpath, spath, direction, equiv_on_hidden,
                      get_first_layer_mask, fanin_rescale, batch_size):
    data_root = os.environ.get("ANYDIM_DATA_ROOT", str(PROJECT_ROOT / "data"))
    dataset_name = "fmnist" if "fmnist" in dpath else "mnist"
    ds = ImageRegressionINRDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        fanin_rescale=fanin_rescale,
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
    parser.add_argument("--init-type", choices=["sp", "mup"], required=True)
    parser.add_argument("--wandb", type=lambda v: v.lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--base-width", type=int, default=16)
    parser.add_argument("--target-widths", type=int, nargs="+", default=[32, 48, 64, 80, 96])
    args = parser.parse_args()

    data_root = Path(os.environ.get("ANYDIM_DATA_ROOT", "data")) / "mnist_inrs"
    ckpt_root = Path(os.environ.get("ANYDIM_DATA_ROOT", "data")) / "checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    set_seed(0)

    if args.init_type == "sp":
        config_path = PROJECT_ROOT / "configs" / "mnist_cls" / "scalegmn_sizegen_regression_sp.yml"
        conditions = {
            "SP baseline": ("sizegen-regression-sp-baseline-full_0p7cqlsf", False),
            "SP dup-equiv": ("sizegen-regression-sp-dupequiv-full_qp6ceiue", True),
        }
    else:
        config_path = PROJECT_ROOT / "configs" / "mnist_cls" / "scalegmn_sizegen_regression_mup.yml"
        conditions = {
            "muP baseline": ("sizegen-regression-mup-baseline-full_rih0g5fa", False),
            "muP dup-equiv": ("sizegen-regression-mup-dupequiv-full_kbjpi4se", True),
        }

    with open(config_path) as f:
        base_conf = yaml.safe_load(f)

    if args.wandb:
        import wandb
        wandb.init(
            project="mnist_cls",
            name=f"sizegen-regression-dup-eval-{args.init_type}",
            tags=["sizegen-regression-dup-eval", args.init_type],
        )

    test_widths = [args.base_width] + args.target_widths

    print(f"{'Condition':<20} | " + " | ".join(f"w{w:>3}" for w in test_widths) + " | mean_dup")
    print("-" * (25 + 10 * len(test_widths) + 12))

    all_results = {}
    for cond_name, (ckpt_name, fanin_rescale) in conditions.items():
        conf = yaml.safe_load(yaml.dump(base_conf))
        conf = overwrite_conf(conf, {})
        conf = resolve_data_paths(conf)

        conf["scalegmn_args"]["layer_layout"] = conf["data"]["train_layer_layouts"][0]

        ckpt_path = ckpt_root / ckpt_name / "best.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] {cond_name}: checkpoint not found at {ckpt_path}")
            continue

        net = build_model_and_load(conf, ckpt_path, device, fanin_rescale)

        direction = conf["scalegmn_args"]["direction"]
        equiv_on_hidden = mask_hidden(conf)
        get_first_layer_mask = mask_input(conf)
        batch_size = conf["batch_size"]
        conf_data = conf["data"]

        psnrs = {}
        mses = {}
        for w in test_widths:
            if w == args.base_width:
                dpath = str(data_root / f"w{w}_{args.init_type}")
                spath = str(data_root / f"w{w}_{args.init_type}" / "mnist_sizegen_splits.json")
            else:
                dpath = str(data_root / f"w{w}_{args.init_type}_dup")
                spath = str(data_root / f"w{w}_{args.init_type}_dup" / "mnist_sizegen_splits.json")

            layout = [2, w, w, 1]
            loader = build_test_loader(
                conf_data, layout, dpath, spath, direction,
                equiv_on_hidden, get_first_layer_mask, fanin_rescale, batch_size
            )
            res = evaluate(net, loader, device)
            psnrs[w] = res["avg_psnr"]
            mses[w] = res["avg_mse"]

        dup_psnrs = [psnrs[w] for w in args.target_widths]
        mean_dup_psnr = sum(dup_psnrs) / len(dup_psnrs)
        psnr_str = " | ".join(f"{psnrs[w]:5.2f}" for w in test_widths)
        print(f"{cond_name:<20} | {psnr_str} | {mean_dup_psnr:5.2f}")

        all_results[cond_name] = {"psnrs": psnrs, "mses": mses}

        if args.wandb:
            import wandb
            for w in test_widths:
                tag = "original" if w == args.base_width else "duplicated"
                wandb.log({
                    f"{cond_name}/{tag}/w{w}_psnr": psnrs[w],
                    f"{cond_name}/{tag}/w{w}_mse": mses[w],
                })
            wandb.log({f"{cond_name}/mean_dup_psnr": mean_dup_psnr})

    print("\n\nSummary (PSNR in dB):")
    print("If duplication equivariance holds, dup-equiv model should get ~same PSNR on all widths.")
    print("Baseline model may degrade on wider duplicated INRs.\n")

    for cond_name, data in all_results.items():
        psnrs = data["psnrs"]
        base_psnr = psnrs.get(args.base_width, 0)
        dup_psnrs = [psnrs[w] for w in args.target_widths if w in psnrs]
        if dup_psnrs:
            mean_dup = sum(dup_psnrs) / len(dup_psnrs)
            max_drop = base_psnr - min(dup_psnrs)
            print(f"  {cond_name}: w16={base_psnr:.2f} dB, mean_dup={mean_dup:.2f} dB, max_drop={max_drop:.2f} dB")

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
