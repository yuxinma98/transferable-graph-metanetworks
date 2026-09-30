"""
Train ScaleGMN on small-width INRs to predict original image pixels (regression).
Evaluate on larger widths for size generalization.

Task: given INR weights as a graph, predict the 28x28 = 784 pixel values of the
original image the INR was fit to. Loss = MSE, metrics = MSE + PSNR.

Usage:
    python scripts/train_sizegen_regression.py --conf configs/mnist_cls/scalegmn_sizegen_regression_sp.yml --wandb True
    python scripts/train_sizegen_regression.py --conf configs/mnist_cls/scalegmn_sizegen_regression_sp.yml --duplication-equiv True --wandb True
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
SCALEGMN_ROOT = PROJECT_ROOT / "src" / "scalegmn"

os.chdir(SCALEGMN_ROOT)
sys.path.insert(0, str(SCALEGMN_ROOT))

import torch_geometric  # noqa: E402
from torch.utils import data as torch_data  # noqa: E402
import wandb  # noqa: E402

from src.scalegmn.models import ScaleGMN  # noqa: E402
from data.image_regression_dataset import ImageRegressionINRDataset  # noqa: E402
from data.duplication_equiv_dataset import FanInLabeledINRDataset  # noqa: E402
from models.layerwise_mean_readout import install_layerwise_mean_readout  # noqa: E402
from models.matrix_product_scale_layer import matrix_product_context  # noqa: E402
from models.bidir_reciprocal import install_bidir_reciprocal  # noqa: E402
from src.utils.helpers import (  # noqa: E402
    count_parameters,
    mask_hidden,
    mask_input,
    overwrite_conf,
    set_seed,
)
from src.utils.optim import setup_optimization  # noqa: E402

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def resolve_data_paths(conf: dict) -> dict:
    """Expand $ANYDIM_DATA_ROOT and make paths absolute."""
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


def build_dataset(layout, dpath, spath, split, debug, direction, equiv_on_hidden,
                  get_first_layer_mask, conf_data, fanin_rescale=False):
    data_root = os.environ.get("ANYDIM_DATA_ROOT", str(PROJECT_ROOT / "data"))
    dataset_name = "fmnist" if "fmnist" in dpath else "mnist"
    return ImageRegressionINRDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        fanin_rescale=fanin_rescale,
        dataset=conf_data["dataset"],
        dataset_path=dpath,
        split_path=spath,
        debug=debug,
        split=split,
        node_pos_embed=conf_data.get("node_pos_embed", False),
        edge_pos_embed=conf_data.get("edge_pos_embed", False),
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=get_first_layer_mask,
        image_size=tuple(conf_data.get("image_size", [28, 28])),
        layer_layout=layout,
        direction=direction,
        switch_to_canon=False,
    )


def build_train_set(conf_data, split, debug, direction, equiv_on_hidden,
                    get_first_layer_mask, fanin_rescale=False):
    datasets = []
    for layout, dpath, spath in zip(
        conf_data["train_layer_layouts"],
        conf_data["train_dataset_paths"],
        conf_data["train_split_paths"],
    ):
        ds = build_dataset(layout, dpath, spath, split, debug, direction,
                          equiv_on_hidden, get_first_layer_mask, conf_data, fanin_rescale)
        datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    return torch_data.ConcatDataset(datasets)


def build_test_sets(conf_data, split, debug, direction, equiv_on_hidden,
                    get_first_layer_mask, fanin_rescale=False):
    """Build one dataset per test width. Returns dict: width -> dataset."""
    test_sets = {}
    for layout, dpath, spath in zip(
        conf_data["test_layer_layouts"],
        conf_data["test_dataset_paths"],
        conf_data["test_split_paths"],
    ):
        width = layout[1]
        ds = build_dataset(layout, dpath, spath, split, debug, direction,
                          equiv_on_hidden, get_first_layer_mask, conf_data, fanin_rescale)
        test_sets[width] = ds
    return test_sets


def mse_to_psnr(mse: float) -> float:
    """Convert MSE (pixel values in [0,1]) to PSNR in dB."""
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
        out = net(batch)  # [B, 784]
        B = batch.num_graphs
        target = batch.label.view(B, -1)  # [B, 784]
        mse = ((out - target) ** 2).mean(dim=-1).sum().item()
        total_mse += mse
        total_n += B
    avg_mse = total_mse / total_n
    return {
        "avg_mse": avg_mse,
        "avg_psnr": mse_to_psnr(avg_mse),
    }


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--conf", required=True)
    parser.add_argument("--wandb", type=lambda v: v.lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--debug", type=lambda v: v.lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--duplication-equiv", type=lambda v: v.lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--direction", type=str, default=None, choices=["forward", "bidirectional"])
    args, _ = parser.parse_known_args()

    conf_path = (PROJECT_ROOT / args.conf).resolve()
    with open(conf_path) as f:
        conf = yaml.safe_load(f)

    conf = overwrite_conf(conf, {})
    conf["wandb"] = args.wandb
    conf["debug"] = args.debug
    conf.setdefault("num_workers", 4)

    if args.lr is not None:
        conf["optimization"]["optimizer_args"]["lr"] = args.lr
    if args.max_steps is not None:
        conf["max_steps"] = args.max_steps
    if args.run_name is not None:
        conf["wandb_args"]["name"] = args.run_name
    if args.direction is not None:
        conf["scalegmn_args"]["direction"] = args.direction

    fanin_rescale = args.duplication_equiv
    if fanin_rescale:
        conf["scalegmn_args"]["gnn_args"]["aggregator"] = "mean"
        conf["scalegmn_args"]["gnn_args"]["msg_equiv_on_hidden"] = True
        print("Duplication-equivariant mode: fan-in weight rescaling + mean aggregation.")

    conf = resolve_data_paths(conf)

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if conf["wandb"]:
        wandb.init(config=conf, **conf["wandb_args"])

    data_root = Path(os.environ.get("ANYDIM_DATA_ROOT", "data"))
    ckpt_root = data_root / "checkpoints"
    run_name = conf.get("wandb_args", {}).get("name", "sizegen-regression")
    if conf["wandb"] and wandb.run is not None:
        ckpt_dir = ckpt_root / f"{run_name}_{wandb.run.id}"
    else:
        ckpt_dir = ckpt_root / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpt_dir}")

    set_seed(conf["train_args"]["seed"])

    # Datasets
    equiv_on_hidden = mask_hidden(conf)
    get_first_layer_mask = mask_input(conf)
    direction = conf["scalegmn_args"]["direction"]

    train_widths = [l[1] for l in conf["data"]["train_layer_layouts"]]
    test_widths = [l[1] for l in conf["data"]["test_layer_layouts"]]
    print(f"Train widths: {train_widths}")
    print(f"Test widths: {test_widths}")

    train_set = build_train_set(conf["data"], "train", conf["debug"], direction,
                                equiv_on_hidden, get_first_layer_mask, fanin_rescale)
    val_set = build_train_set(conf["data"], "val", conf["debug"], direction,
                              equiv_on_hidden, get_first_layer_mask, fanin_rescale)

    # Per-width test sets
    test_sets = build_test_sets(conf["data"], "test", conf["debug"], direction,
                                equiv_on_hidden, get_first_layer_mask, fanin_rescale)

    print(f"Train: {len(train_set):,}  Val: {len(val_set):,}")
    for w, ds in sorted(test_sets.items()):
        print(f"  Test w={w}: {len(ds):,}")

    # Use smallest train width for model init
    conf["scalegmn_args"]["layer_layout"] = conf["data"]["train_layer_layouts"][0]

    train_loader = torch_geometric.loader.DataLoader(
        train_set, batch_size=conf["batch_size"], shuffle=True,
        num_workers=conf["num_workers"], pin_memory=True,
    )
    val_loader = torch_geometric.loader.DataLoader(
        val_set, batch_size=conf["batch_size"], shuffle=False,
    )
    test_loaders = {
        w: torch_geometric.loader.DataLoader(ds, batch_size=conf["batch_size"], shuffle=False,
                                              num_workers=conf["num_workers"], pin_memory=True)
        for w, ds in sorted(test_sets.items())
    }

    # Model (the context manager is a no-op unless a matrix-product MSG is configured)
    with matrix_product_context(conf["scalegmn_args"]):
        net = ScaleGMN(conf["scalegmn_args"]).to(device)
    if fanin_rescale:
        install_layerwise_mean_readout(net, conf["scalegmn_args"])
        print("Installed layer-wise mean readout for graph-level duplication equivariance.")
    if install_bidir_reciprocal(net, conf["scalegmn_args"]):
        print("Installed reciprocal backward edge features (bidirectional scale equivariance).")
    print(net)
    cnt_p = count_parameters(net=net)
    if conf["wandb"]:
        wandb.log({"number_of_parameters": cnt_p}, step=0)

    # Optimization
    criterion = torch.nn.MSELoss()
    conf_opt = conf["optimization"]
    optimizer, scheduler = setup_optimization(
        [p for p in net.parameters() if p.requires_grad],
        optimizer_name=conf_opt["optimizer_name"],
        optimizer_args=conf_opt["optimizer_args"],
        scheduler_args=conf_opt["scheduler_args"],
    )

    # Training loop
    best_val_mse = float("inf")
    best_test_results = {}
    patience = conf["train_args"]["patience"]
    val_mse_history = []
    max_steps = conf.get("max_steps", None)
    global_step = 0
    done = False

    for epoch in range(conf["train_args"]["num_epochs"]):
        if done:
            break
        net.train()
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            if max_steps is not None and global_step >= max_steps:
                done = True
                break
            batch = batch.to(device)
            optimizer.zero_grad()
            out = net(batch)  # [B, 784]
            target = batch.label.view(batch.num_graphs, -1)  # [B, 784]
            loss = criterion(out, target)
            loss.backward()

            log = {}
            if conf["optimization"]["clip_grad"]:
                log["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), conf["optimization"]["clip_grad_max_norm"]
                )
            optimizer.step()

            if scheduler[1] not in (None, "ReduceLROnPlateau"):
                scheduler[0].step()
                log["lr"] = scheduler[0].get_last_lr()[0]

            log["train/mse"] = loss.item()
            log["train/psnr"] = mse_to_psnr(loss.item())
            log["epoch"] = epoch
            if conf["wandb"]:
                wandb.log(log, step=global_step)
            global_step += 1

        # Validation
        val_res = evaluate(net, val_loader, device)
        val_mse = val_res["avg_mse"]
        val_mse_history.append(val_mse)

        # Per-width test evaluation
        test_results = {}
        for w, loader in sorted(test_loaders.items()):
            test_results[w] = evaluate(net, loader, device)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_test_results = {w: r.copy() for w, r in test_results.items()}
            torch.save({
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mse": val_mse,
                "test_results": {w: r for w, r in test_results.items()},
            }, ckpt_dir / "best.pt")

        # Print results
        in_dist = [w for w in sorted(test_results.keys()) if w in train_widths]
        out_dist = [w for w in sorted(test_results.keys()) if w not in train_widths]
        in_str = "  ".join(f"w{w}={test_results[w]['avg_psnr']:.1f}dB" for w in in_dist)
        out_str = "  ".join(f"w{w}={test_results[w]['avg_psnr']:.1f}dB" for w in out_dist)
        print(
            f"Epoch {epoch:3d}  val_mse={val_mse:.6f} val_psnr={val_res['avg_psnr']:.1f}dB  "
            f"IN: {in_str}  OUT: {out_str}"
        )

        if conf["wandb"]:
            log = {
                "val/mse": val_mse,
                "val/psnr": val_res["avg_psnr"],
                "val/best_mse": best_val_mse,
                "epoch": epoch,
            }
            for w, res in test_results.items():
                tag = "in_dist" if w in train_widths else "out_dist"
                log[f"test/{tag}/w{w}_mse"] = res["avg_mse"]
                log[f"test/{tag}/w{w}_psnr"] = res["avg_psnr"]
            log["test/in_dist/mean_mse"] = sum(test_results[w]["avg_mse"] for w in in_dist) / len(in_dist) if in_dist else 0
            log["test/out_dist/mean_mse"] = sum(test_results[w]["avg_mse"] for w in out_dist) / len(out_dist) if out_dist else 0
            log["test/in_dist/mean_psnr"] = sum(test_results[w]["avg_psnr"] for w in in_dist) / len(in_dist) if in_dist else 0
            log["test/out_dist/mean_psnr"] = sum(test_results[w]["avg_psnr"] for w in out_dist) / len(out_dist) if out_dist else 0
            wandb.log(log, step=global_step)

        # Early stopping (lower MSE is better)
        if len(val_mse_history) > patience:
            if min(val_mse_history[-patience:]) >= val_mse_history[-patience - 1]:
                print(f"Early stopping: no improvement in last {patience} epochs.")
                break

    print(f"\nBest val MSE: {best_val_mse:.6f}  PSNR: {mse_to_psnr(best_val_mse):.1f} dB")
    print("Test MSE/PSNR @ best val:")
    for w in sorted(best_test_results.keys()):
        tag = "IN" if w in train_widths else "OUT"
        r = best_test_results[w]
        print(f"  w={w:3d} ({tag}): MSE={r['avg_mse']:.6f}  PSNR={r['avg_psnr']:.1f} dB")

    if conf["wandb"]:
        wandb.summary["best_val_mse"] = best_val_mse
        wandb.summary["best_val_psnr"] = mse_to_psnr(best_val_mse)
        for w, r in best_test_results.items():
            tag = "in_dist" if w in train_widths else "out_dist"
            wandb.summary[f"best_test/{tag}/w{w}_mse"] = r["avg_mse"]
            wandb.summary[f"best_test/{tag}/w{w}_psnr"] = r["avg_psnr"]
        wandb.finish()


if __name__ == "__main__":
    main()
