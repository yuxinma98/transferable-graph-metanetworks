"""
Size generalization for CNN accuracy prediction: train on one channel width, test on many.

Fork of ``scripts/train_sizegen.py`` with the classification head/metrics replaced by
the regression ones from ``src/scalegmn/predicting_generalization.py``:

* head ``sigmoid(net(batch))``, MSE loss against the CNN's test accuracy;
* per-width metrics ``{Kendall tau, R2, L1, R2_recal}``;
* **model selection on val tau at the train width only** — no OOD width may
  influence which checkpoint is kept.

Metric notes
------------
``tau`` (Kendall) is the headline number: it is invariant to any monotone per-width
recalibration, so it separates "can the metanetwork read wide weights" from "is its
output offset at this width".  ``R2`` and ``L1`` are the calibration-*sensitive*
counterparts.  ``R2_recal`` is a deliberately cheating diagnostic: it fits the best
affine map ``a*pred + b`` **using the test labels**, so it upper-bounds what any
per-width recalibration could achieve.  Reading the pair:

    R2 low  + R2_recal high  ->  pure calibration failure (the response shift)
    R2 low  + R2_recal low   ->  genuine failure to read wide weights

Readout
-------
Both the baseline and the dup-equiv conditions install a readout from *our* repo,
because upstream's ``readout_range: last_layer`` path reshapes with the train width's
``num_nodes`` and crashes at OOD widths.  Default ``--readout last_layer``
(:class:`src.models.last_layer_readout.LastLayerReadout`) matches the upstream config
for this task and is duplication-invariant by construction, since the 10 output nodes
are never widened.  ``--readout layerwise_mean`` is the INR-side alternative.

Usage
-----
    python scripts/train_sizegen_predgen.py --conf configs/cifar10_predgen/scalegmn_sizegen_sp_v1.yml --wandb True
    python scripts/train_sizegen_predgen.py --conf configs/cifar10_predgen/scalegmn_sizegen_sp_v1.yml \
        --duplication-equiv True --lr 5e-4 --wandb True

    # Score an existing checkpoint on the config's test widths, without training:
    python scripts/train_sizegen_predgen.py --conf configs/cifar10_predgen/scalegmn_sizegen_sp_v1.yml \
        --eval-only-ckpt $ANYDIM_DATA_ROOT/checkpoints/<run>/best.pt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
SCALEGMN_ROOT = PROJECT_ROOT / "src" / "scalegmn"

os.chdir(SCALEGMN_ROOT)
sys.path.insert(0, str(SCALEGMN_ROOT))

import torch_geometric  # noqa: E402
import wandb  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from torch.utils import data as torch_data  # noqa: E402

from src.scalegmn.models import ScaleGMN  # noqa: E402
from data.cnn_zoo_dataset import CNNZooDataset, FanInCNNZooDataset  # noqa: E402
from models.matrix_product_scale_layer import matrix_product_context  # noqa: E402
from models.last_layer_readout import install_last_layer_readout  # noqa: E402
from models.layerwise_mean_readout import install_layerwise_mean_readout  # noqa: E402
from models.bidir_reciprocal import install_bidir_reciprocal  # noqa: E402
from src.utils.helpers import (  # noqa: E402
    count_parameters,
    mask_hidden,
    mask_input,
    overwrite_conf,
    set_seed,
)
from src.utils.loss import select_criterion  # noqa: E402
from src.utils.optim import setup_optimization  # noqa: E402

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

PATH_KEYS = ("train_dataset_paths", "train_split_paths",
             "test_dataset_paths", "test_split_paths")


def resolve_data_paths(conf: dict) -> dict:
    """Expand $ANYDIM_DATA_ROOT and make paths absolute."""
    data = conf.get("data", {})
    for key in PATH_KEYS:
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
    dataset_cls = FanInCNNZooDataset if fanin_rescale else CNNZooDataset
    return dataset_cls(
        dataset=conf_data["dataset"],
        dataset_path=dpath,
        split_path=spath,
        debug=debug,
        split=split,
        node_pos_embed=conf_data.get("node_pos_embed", False),
        edge_pos_embed=conf_data.get("edge_pos_embed", False),
        equiv_on_hidden=equiv_on_hidden,
        get_first_layer_mask=get_first_layer_mask,
        layer_layout=layout,
        direction=direction,
        activation_function=conf_data.get("activation_function", "relu"),
        max_kernel_size=tuple(conf_data.get("max_kernel_size", [3, 3])),
        max_items=conf_data.get("max_items_per_split"),
    )


def build_train_set(conf_data, split, debug, direction, equiv_on_hidden,
                    get_first_layer_mask, fanin_rescale=False):
    datasets = [
        build_dataset(layout, dpath, spath, split, debug, direction,
                      equiv_on_hidden, get_first_layer_mask, conf_data, fanin_rescale)
        for layout, dpath, spath in zip(
            conf_data["train_layer_layouts"],
            conf_data["train_dataset_paths"],
            conf_data["train_split_paths"],
        )
    ]
    return datasets[0] if len(datasets) == 1 else torch_data.ConcatDataset(datasets)


def build_test_sets(conf_data, split, debug, direction, equiv_on_hidden,
                    get_first_layer_mask, fanin_rescale=False):
    """One dataset per test width. Returns ``{width: dataset}``."""
    return {
        layout[1]: build_dataset(layout, dpath, spath, split, debug, direction,
                                 equiv_on_hidden, get_first_layer_mask, conf_data,
                                 fanin_rescale)
        for layout, dpath, spath in zip(
            conf_data["test_layer_layouts"],
            conf_data["test_dataset_paths"],
            conf_data["test_split_paths"],
        )
    }


def regression_metrics(actual: np.ndarray, pred: np.ndarray, loss: float) -> dict:
    """Kendall tau / R2 / L1 / oracle-recalibrated R2 for one width."""
    l1 = float(np.abs(pred - actual).mean())
    tau = kendalltau(actual, pred).correlation
    rsq = float(r2_score(actual, pred))
    # best affine recalibration fitted ON the test labels (a cheating upper bound).
    # The R2 of the optimal a*pred + b IS the squared Pearson correlation, so take
    # that directly rather than least-squares-fitting (a, b), which is ill-conditioned
    # exactly when it matters least — when the predictions have collapsed to a point.
    if pred.std() > 0 and actual.std() > 0:
        rsq_recal = float(np.corrcoef(actual, pred)[0, 1] ** 2)
    else:
        rsq_recal = float("nan")
    return {"loss": loss, "l1": l1, "tau": float(tau), "rsq": rsq,
            "rsq_recal": rsq_recal, "pred_mean": float(pred.mean()),
            "actual_mean": float(actual.mean())}


@torch.no_grad()
def evaluate(net, loader, criterion, device):
    net.eval()
    preds, actuals, losses, ns = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        target = batch.y.float()
        pred = F.sigmoid(net(batch)).squeeze(-1)
        losses.append(criterion(pred, target).item() * batch.num_graphs)
        ns.append(batch.num_graphs)
        preds.append(pred.detach().cpu().numpy())
        actuals.append(target.cpu().numpy())
    pred = np.concatenate(preds)
    actual = np.concatenate(actuals)
    return regression_metrics(actual, pred, sum(losses) / sum(ns)), pred, actual


def fmt(res: dict) -> str:
    return (f"tau={res['tau']:.4f} R2={res['rsq']:+.3f} L1={res['l1']:.4f} "
            f"R2rc={res['rsq_recal']:+.3f}")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--conf", required=True)
    bool_arg = lambda v: v.lower() in ("true", "1", "yes")  # noqa: E731
    parser.add_argument("--wandb", type=bool_arg, default=False)
    parser.add_argument("--debug", type=bool_arg, default=False)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--duplication-equiv", type=bool_arg, default=False)
    parser.add_argument("--direction", type=str, default=None,
                        choices=["forward", "bidirectional"])
    parser.add_argument("--readout", type=str, default="last_layer",
                        choices=["last_layer", "layerwise_mean"],
                        help="width-agnostic readout to install (see module docstring)")
    parser.add_argument("--bidir-reciprocal", type=bool_arg, default=True,
                        help="Feed 1/W to the backward layers, as the config's "
                             "`reciprocal` flag asks. This is part of the definition of a "
                             "bidirectional scale-equivariant model, not an option; set "
                             "False only to reproduce runs from before the fix.")
    parser.add_argument("--eval-only-ckpt", type=str, default=None,
                        help="Skip training: load this checkpoint and report per-width "
                             "metrics. Used to score checkpoints on extra test widths.")
    parser.add_argument("--save-predictions", type=str, default=None,
                        help="Optional .npz path for per-width (pred, actual) arrays.")
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

    eval_only = args.eval_only_ckpt is not None
    fanin_rescale = args.duplication_equiv
    if fanin_rescale:
        conf["scalegmn_args"]["gnn_args"]["aggregator"] = "mean"
        conf["scalegmn_args"]["gnn_args"]["msg_equiv_on_hidden"] = True
        print("Duplication-equivariant mode: fan-in kernel rescaling + mean aggregation.")
    # our readouts are installed into the full_graph slot; see install_* docstrings
    conf["scalegmn_args"]["readout_range"] = "full_graph"

    conf = resolve_data_paths(conf)

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if conf["wandb"]:
        wandb.init(config=conf, **conf["wandb_args"])

    data_root = Path(os.environ.get("ANYDIM_DATA_ROOT", "data"))
    ckpt_root = data_root / "checkpoints"
    run_name = conf.get("wandb_args", {}).get("name", "sizegen-predgen")
    ckpt_dir = ckpt_root / (f"{run_name}_{wandb.run.id}"
                            if conf["wandb"] and wandb.run is not None else run_name)
    if not eval_only:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints will be saved to: {ckpt_dir}")

    set_seed(conf["train_args"]["seed"])

    equiv_on_hidden = mask_hidden(conf)
    get_first_layer_mask = mask_input(conf)
    direction = conf["scalegmn_args"]["direction"]

    train_widths = [l[1] for l in conf["data"]["train_layer_layouts"]]
    test_widths = [l[1] for l in conf["data"]["test_layer_layouts"]]
    print(f"Train widths: {train_widths}")
    print(f"Test widths: {test_widths}")

    if not eval_only:
        train_set = build_train_set(conf["data"], "train", conf["debug"], direction,
                                    equiv_on_hidden, get_first_layer_mask, fanin_rescale)
        val_set = build_train_set(conf["data"], "val", conf["debug"], direction,
                                  equiv_on_hidden, get_first_layer_mask, fanin_rescale)
        print(f"Train: {len(train_set):,}  Val: {len(val_set):,}")

    test_sets = build_test_sets(conf["data"], "test", conf["debug"], direction,
                                equiv_on_hidden, get_first_layer_mask, fanin_rescale)
    for w, ds in sorted(test_sets.items()):
        print(f"  Test w={w}: {len(ds):,}")

    conf["scalegmn_args"]["layer_layout"] = conf["data"]["train_layer_layouts"][0]

    if not eval_only:
        train_loader = torch_geometric.loader.DataLoader(
            train_set, batch_size=conf["batch_size"], shuffle=True,
            num_workers=conf["num_workers"], pin_memory=True)
        val_loader = torch_geometric.loader.DataLoader(
            val_set, batch_size=conf["batch_size"], shuffle=False,
            num_workers=conf["num_workers"], pin_memory=True)
    train_width = train_widths[0]
    test_loaders = {
        w: torch_geometric.loader.DataLoader(
            ds,
            # keep activation memory flat: the graph has O(w^2) edges
            batch_size=max(1, conf["batch_size"] * train_width ** 2 // w ** 2),
            shuffle=False, num_workers=conf["num_workers"], pin_memory=True)
        for w, ds in sorted(test_sets.items())
    }

    # Model
    # the context manager is a no-op unless gnn_args.message_fn_type is
    # matrix_product_conv, in which case it swaps in the conv matrix-product MSG
    # (src/models/conv_matrix_product_layer.py) without editing the git subtree
    with matrix_product_context(conf["scalegmn_args"]):
        net = ScaleGMN(conf["scalegmn_args"]).to(device)
    if args.readout == "last_layer":
        install_last_layer_readout(net, conf["scalegmn_args"])
    else:
        install_layerwise_mean_readout(net, conf["scalegmn_args"])
    # A bidirectional `symmetry: scale` model is *defined* with the reciprocal backward
    # edge feature 1/W[u, v] -- without it the backward messages carry lam_v^2/lam_u and
    # the model is not scale-equivariant at all. Upstream assigns its own `reciprocal`
    # flag and never reads it; this installs what the flag asks for (works on conv graphs
    # under the 1/0 := 0 convention, see src/models/bidir_reciprocal.py).
    if args.bidir_reciprocal and install_bidir_reciprocal(net, conf["scalegmn_args"]):
        print("Installed reciprocal backward edge features "
              "(bidirectional scale equivariance).")
    net = net.to(device)
    print(net)
    print(f"readout: {args.readout}  (width-agnostic)")
    cnt_p = count_parameters(net=net)
    if conf["wandb"]:
        wandb.log({"number_of_parameters": cnt_p}, step=0)

    criterion = select_criterion(conf["train_args"]["loss"], {})

    def eval_all_widths(saved_preds=None):
        results = {}
        for w, loader in sorted(test_loaders.items()):
            res, pred, actual = evaluate(net, loader, criterion, device)
            results[w] = res
            if saved_preds is not None:
                saved_preds[f"w{w}_pred"] = pred
                saved_preds[f"w{w}_actual"] = actual
        return results

    if eval_only:
        ckpt = torch.load(args.eval_only_ckpt, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"])
        print(f"\nLoaded {args.eval_only_ckpt} (epoch {ckpt.get('epoch', '?')}, "
              f"val_tau {ckpt.get('val_tau', float('nan')):.4f})")
        saved = {} if args.save_predictions else None
        results = eval_all_widths(saved)
        print("Per-width test metrics:")
        for w, res in sorted(results.items()):
            tag = "IN " if w in train_widths else "OUT"
            print(f"  w={w:5d} ({tag}): {fmt(res)}", flush=True)
            if conf["wandb"]:
                scope = "in_dist" if w in train_widths else "out_dist"
                for key, val in res.items():
                    wandb.summary[f"eval_test/{scope}/w{w}_{key}"] = val
        ood = [w for w in sorted(results) if w not in train_widths]
        if ood:
            mean_tau = sum(results[w]["tau"] for w in ood) / len(ood)
            print(f"  mean OOD tau: {mean_tau:.4f}")
            if conf["wandb"]:
                wandb.summary["eval_test/out_dist/mean_tau"] = mean_tau
        if saved is not None:
            np.savez(args.save_predictions, **saved)
            print(f"  predictions -> {args.save_predictions}")
        if conf["wandb"]:
            wandb.finish()
        return

    conf_opt = conf["optimization"]
    optimizer, scheduler = setup_optimization(
        [p for p in net.parameters() if p.requires_grad],
        optimizer_name=conf_opt["optimizer_name"],
        optimizer_args=conf_opt["optimizer_args"],
        scheduler_args=conf_opt["scheduler_args"],
    )

    best_val_tau = -float("inf")
    best_test_results = {}
    patience = conf["train_args"].get("patience")
    val_tau_history = []
    max_steps = conf.get("max_steps")
    global_step = 0
    done = False

    for epoch in range(conf["train_args"]["num_epochs"]):
        if done:
            break
        net.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            if max_steps is not None and global_step >= max_steps:
                done = True
                break
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = F.sigmoid(net(batch)).squeeze(-1)
            loss = criterion(pred, batch.y.float())
            loss.backward()

            log = {}
            if conf["optimization"]["clip_grad"]:
                log["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), conf["optimization"]["clip_grad_max_norm"])
            optimizer.step()

            if scheduler[1] not in (None, "ReduceLROnPlateau"):
                scheduler[0].step()
                log["lr"] = scheduler[0].get_last_lr()[0]

            log[f"train/{conf['train_args']['loss']}"] = loss.item()
            log["epoch"] = epoch
            if conf["wandb"]:
                wandb.log(log, step=global_step)
            global_step += 1

        # Validation at the train width only — this is what selects the checkpoint.
        val_res, _, _ = evaluate(net, val_loader, criterion, device)
        val_tau = val_res["tau"]
        val_tau_history.append(val_tau)

        test_results = eval_all_widths()

        if val_tau > best_val_tau:
            best_val_tau = val_tau
            best_test_results = {w: dict(r) for w, r in test_results.items()}
            torch.save({
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_tau": val_tau,
                "val_results": val_res,
                "test_results": test_results,
                "readout": args.readout,
                "duplication_equiv": fanin_rescale,
            }, ckpt_dir / "best.pt")

        in_dist = [w for w in sorted(test_results) if w in train_widths]
        out_dist = [w for w in sorted(test_results) if w not in train_widths]
        in_str = "  ".join(f"w{w}:{test_results[w]['tau']:.4f}" for w in in_dist)
        out_str = "  ".join(f"w{w}:{test_results[w]['tau']:.4f}" for w in out_dist)
        print(f"Epoch {epoch:3d}  val_tau={val_tau:.4f}  "
              f"IN tau: {in_str}  OUT tau: {out_str}", flush=True)

        if conf["wandb"]:
            log = {"val/tau": val_tau, "val/best_tau": best_val_tau,
                   "val/rsq": val_res["rsq"], "val/l1": val_res["l1"],
                   "val/loss": val_res["loss"], "epoch": epoch}
            for w, res in test_results.items():
                tag = "in_dist" if w in train_widths else "out_dist"
                for key in ("tau", "rsq", "l1", "rsq_recal", "loss"):
                    log[f"test/{tag}/w{w}_{key}"] = res[key]
            if out_dist:
                log["test/out_dist/mean_tau"] = sum(
                    test_results[w]["tau"] for w in out_dist) / len(out_dist)
                log["test/out_dist/mean_rsq"] = sum(
                    test_results[w]["rsq"] for w in out_dist) / len(out_dist)
            wandb.log(log, step=global_step)

        if patience and len(val_tau_history) > patience:
            if max(val_tau_history[-patience:]) <= val_tau_history[-patience - 1]:
                print(f"Early stopping: no improvement in last {patience} epochs.")
                break

    print(f"\nBest val tau: {best_val_tau:.4f}")
    print("Test metrics @ best val tau:")
    for w, res in sorted(best_test_results.items()):
        tag = "IN " if w in train_widths else "OUT"
        print(f"  w={w:5d} ({tag}): {fmt(res)}")

    if conf["wandb"]:
        wandb.summary["best_val_tau"] = best_val_tau
        for w, res in best_test_results.items():
            tag = "in_dist" if w in train_widths else "out_dist"
            for key, val in res.items():
                wandb.summary[f"best_test/{tag}/w{w}_{key}"] = val
        ood = [w for w in best_test_results if w not in train_widths]
        if ood:
            wandb.summary["best_test/out_dist/mean_tau"] = sum(
                best_test_results[w]["tau"] for w in ood) / len(ood)
        wandb.finish()


if __name__ == "__main__":
    main()
