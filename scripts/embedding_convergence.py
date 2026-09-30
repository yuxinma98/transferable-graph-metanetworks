"""
Empirical investigation of embedding distribution convergence.

Extracts graph-level embeddings from trained/random GNN models across widths,
computes distributional distances (Sliced Wasserstein, MMD), and visualizes
whether embedding distributions converge as width increases.

Usage:
    python scripts/embedding_convergence.py \
        --init-type mup --model-type gmn \
        --widths 16 24 32 48 64 80 96 \
        --n-random-seeds 10 --output-dir outputs/embedding_convergence

    # With trained checkpoints:
    python scripts/embedding_convergence.py \
        --init-type mup --model-type gmn \
        --ckpt-dupequiv /path/to/best.pt \
        --ckpt-baseline /path/to/best.pt \
        --widths 16 24 32 48 64 80 96 \
        --output-dir outputs/embedding_convergence
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import wasserstein_distance
from tqdm import tqdm

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


def build_model(conf, ckpt_path, device, fanin_rescale, seed=None):
    """Build a ScaleGMN model, optionally loading from checkpoint."""
    if seed is not None:
        set_seed(seed)

    if fanin_rescale:
        conf["scalegmn_args"]["gnn_args"]["aggregator"] = "mean"

    net = ScaleGMN(conf["scalegmn_args"]).to(device)
    if fanin_rescale:
        install_layerwise_mean_readout(net, conf["scalegmn_args"])

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"])

    net.eval()
    return net


def build_model_2d(conf, device, fanin_rescale, seed):
    """Build a model with d_out=2 for 2D point cloud visualization."""
    conf = yaml.safe_load(yaml.dump(conf))
    conf["scalegmn_args"]["readout_args"]["d_out"] = 2
    return build_model(conf, None, device, fanin_rescale, seed=seed)


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


@torch.no_grad()
def extract_embeddings(net, loader, device):
    """Extract graph-level embeddings using a forward hook on global_mlp."""
    net.eval()
    embeddings = []
    labels = []

    captured = {}

    def hook_fn(module, input, output):
        captured["emb"] = input[0].detach().cpu()

    readout = net.gnn.readout
    handle = readout.global_mlp.register_forward_hook(hook_fn)

    for batch in loader:
        batch = batch.to(device)
        _ = net(batch)
        embeddings.append(captured["emb"])
        labels.append(batch.label.cpu())

    handle.remove()
    return torch.cat(embeddings, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


@torch.no_grad()
def extract_2d_outputs(net, loader, device):
    """Extract 2D output logits for point cloud visualization."""
    net.eval()
    points = []
    labels = []

    for batch in loader:
        batch = batch.to(device)
        out = net(batch)
        points.append(out.cpu())
        labels.append(batch.label.cpu())

    return torch.cat(points, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def sliced_wasserstein_distance(X, Y, n_projections=1000, seed=0):
    """Compute sliced Wasserstein distance between two point clouds."""
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    directions = rng.randn(n_projections, d)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    proj_X = X @ directions.T  # [N, n_proj]
    proj_Y = Y @ directions.T  # [M, n_proj]

    distances = np.array([
        wasserstein_distance(proj_X[:, i], proj_Y[:, i])
        for i in range(n_projections)
    ])
    return distances.mean()


def mmd_gaussian(X, Y, bandwidth=None, max_samples=500, seed=0):
    """Compute MMD with Gaussian kernel (median heuristic for bandwidth).

    Uses cdist for efficient pairwise distance computation.
    """
    from scipy.spatial.distance import cdist
    rng = np.random.RandomState(seed)

    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    XY = np.vstack([X, Y])
    if bandwidth is None:
        dists_sq = cdist(XY, XY, metric='sqeuclidean')
        bandwidth = np.median(dists_sq[np.triu_indices(len(XY), k=1)]) + 1e-8

    Kxx = np.exp(-cdist(X, X, metric='sqeuclidean') / (2 * bandwidth))
    Kyy = np.exp(-cdist(Y, Y, metric='sqeuclidean') / (2 * bandwidth))
    Kxy = np.exp(-cdist(X, Y, metric='sqeuclidean') / (2 * bandwidth))

    n, m = len(X), len(Y)
    mmd2 = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + \
            (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - \
            2 * Kxy.mean()
    return max(0, mmd2) ** 0.5


def compute_pairwise_distances(embeddings_by_width, widths, n_projections=1000):
    """Compute pairwise SWD and MMD matrices."""
    n = len(widths)
    swd_matrix = np.zeros((n, n))
    mmd_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            X = embeddings_by_width[widths[i]]
            Y = embeddings_by_width[widths[j]]
            swd = sliced_wasserstein_distance(X, Y, n_projections=n_projections)
            mmd = mmd_gaussian(X, Y)
            swd_matrix[i, j] = swd_matrix[j, i] = swd
            mmd_matrix[i, j] = mmd_matrix[j, i] = mmd

    return swd_matrix, mmd_matrix


def plot_convergence_curves(results, widths, plot_dir, metric="swd"):
    """Plot distance-to-max-width curves for all conditions."""
    import matplotlib.pyplot as plt

    ref_idx = len(widths) - 1  # largest width as reference
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for cond_name, data in results.items():
        if "random" in cond_name:
            matrices = data["matrices"]  # list of matrices per seed
            dists = np.array([m[:, ref_idx] for m in matrices])  # [K, n_widths]
            mean_d = dists.mean(axis=0)
            std_d = dists.std(axis=0)
            xs = widths[:-1]
            ax.plot(xs, mean_d[:-1], 'o-', label=cond_name)
            ax.fill_between(xs, (mean_d - std_d)[:-1], (mean_d + std_d)[:-1], alpha=0.2)
        else:
            m = data["matrices"][0]
            dists = m[:, ref_idx]
            xs = widths[:-1]
            ax.plot(xs, dists[:-1], 's--', label=cond_name)

    ax.set_xlabel("Width")
    ax.set_ylabel(f"{metric.upper()} to w{widths[-1]}")
    ax.set_title(f"Embedding convergence ({metric.upper()})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f"convergence_{metric}_overlay.pdf", dpi=150)
    plt.close()


def plot_heatmap(matrix, widths, title, path):
    """Plot a distance heatmap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(len(widths)))
    ax.set_yticks(range(len(widths)))
    ax.set_xticklabels([f"w{w}" for w in widths])
    ax.set_yticklabels([f"w{w}" for w in widths])
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_2d_pointclouds(points_by_width, labels_by_width, widths, output_path,
                        title="2D embeddings"):
    """Scatter plot of 2D outputs colored by width."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(widths))

    for i, w in enumerate(widths):
        pts = points_by_width[w]
        ax.scatter(pts[:, 0], pts[:, 1], c=[cmap(i)], s=3, alpha=0.3, label=f"w{w}")

    ax.legend(markerscale=5)
    ax.set_title(title)
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_2d_grid(all_seeds_points, widths, output_path, ncols=3):
    """Grid of 2D point cloud panels, one per seed."""
    import matplotlib.pyplot as plt

    n_seeds = len(all_seeds_points)
    nrows = (n_seeds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    cmap = plt.colormaps.get_cmap("tab10").resampled(len(widths))

    for seed_idx, points_by_width in enumerate(all_seeds_points):
        ax = axes[seed_idx]
        for i, w in enumerate(widths):
            pts = points_by_width[w]
            ax.scatter(pts[:, 0], pts[:, 1], c=[cmap(i)], s=2, alpha=0.3,
                       label=f"w{w}" if seed_idx == 0 else None)
        ax.set_title(f"seed {seed_idx}")
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_seeds, len(axes)):
        axes[idx].set_visible(False)

    if n_seeds > 0:
        handles, lbls = axes[0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc='upper right', markerscale=5)

    plt.suptitle("2D random projections by width")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_perclass_summary(perclass_swd, widths, output_path):
    """Plot per-class SWD to max width."""
    import matplotlib.pyplot as plt

    ref_idx = len(widths) - 1
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for cls in range(10):
        dists = perclass_swd[cls][:, ref_idx]
        ax.plot(widths[:-1], dists[:-1], 'o-', label=f"digit {cls}", alpha=0.7)

    ax.set_xlabel("Width")
    ax.set_ylabel(f"SWD to w{widths[-1]}")
    ax.set_title("Per-class SWD convergence (dup-equiv, seed 0)")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-type", choices=["sp", "mup"], required=True)
    parser.add_argument("--model-type", choices=["scalegmn", "gmn"], default="gmn")
    parser.add_argument("--ckpt-dupequiv", type=str, default=None)
    parser.add_argument("--ckpt-baseline", type=str, default=None)
    parser.add_argument("--widths", type=int, nargs="+", default=[16, 24, 32, 48, 64, 80, 96])
    parser.add_argument("--n-random-seeds", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs/embedding_convergence")
    parser.add_argument("--n-projections", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    plot_dir = (PROJECT_ROOT / args.output_dir).resolve() / f"{args.model_type}_{args.init_type}" / "plots"
    data_root_dir = Path(os.environ.get("ANYDIM_DATA_ROOT", "data")) / "embedding_convergence" / f"{args.model_type}_{args.init_type}"
    output_dir = data_root_dir  # embeddings/distances go here
    (output_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    (output_dir / "embeddings_2d").mkdir(parents=True, exist_ok=True)
    (output_dir / "distances").mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(os.environ.get("ANYDIM_DATA_ROOT", "data")) / "mnist_inrs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    # Load config
    config_path = PROJECT_ROOT / "configs" / "mnist_cls" / f"{args.model_type}_sizegen_{args.init_type}.yml"
    with open(config_path) as f:
        base_conf = yaml.safe_load(f)

    # --- Embedding extraction ---
    conditions = []
    if args.ckpt_dupequiv:
        conditions.append(("trained-dupequiv", args.ckpt_dupequiv, True, [0]))
    if args.ckpt_baseline:
        conditions.append(("trained-baseline", args.ckpt_baseline, False, [0]))
    conditions.append(("random-dupequiv", None, True, list(range(args.n_random_seeds))))
    conditions.append(("random-baseline", None, False, list(range(args.n_random_seeds))))

    all_results = {}  # {cond_name: {"matrices": [swd_matrices], ...}}

    for cond_name, ckpt_path, fanin_rescale, seeds in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} (fanin_rescale={fanin_rescale}, seeds={len(seeds)})")
        print(f"{'='*60}")

        swd_matrices = []
        mmd_matrices = []

        for seed in tqdm(seeds, desc=f"{cond_name} seeds"):
            embeddings_by_width = {}
            labels_by_width = {}
            all_cached = True

            for w in args.widths:
                if "trained" in cond_name:
                    cache_file = output_dir / "embeddings" / f"emb_{cond_name}_w{w}.npz"
                else:
                    cache_file = output_dir / "embeddings" / f"emb_{cond_name}_seed{seed}_w{w}.npz"

                if cache_file.exists() and not args.recompute:
                    data = np.load(cache_file)
                    embeddings_by_width[w] = data["embeddings"]
                    labels_by_width[w] = data["labels"]
                else:
                    all_cached = False
                    break

            if all_cached:
                print(f"  seed {seed}: loaded from cache")
            else:
                # Build model
                conf = yaml.safe_load(yaml.dump(base_conf))
                conf = overwrite_conf(conf, {})
                conf = resolve_data_paths(conf)
                if fanin_rescale:
                    conf["scalegmn_args"]["gnn_args"]["msg_equiv_on_hidden"] = True
                conf["scalegmn_args"]["layer_layout"] = conf["data"]["train_layer_layouts"][0]

                net = build_model(conf, ckpt_path, device, fanin_rescale,
                                  seed=seed if ckpt_path is None else None)

                # Warmup for lazy global_mlp init
                direction = conf["scalegmn_args"]["direction"]
                equiv_on_hidden = mask_hidden(conf)
                get_first_layer_mask = mask_input(conf)
                conf_data = conf["data"]

                warmup_loader = build_test_loader(
                    conf_data, conf["data"]["train_layer_layouts"][0],
                    conf["data"]["train_dataset_paths"][0],
                    conf["data"]["train_split_paths"][0],
                    direction, equiv_on_hidden, get_first_layer_mask,
                    fanin_rescale, batch_size=4
                )
                warmup_batch = next(iter(warmup_loader)).to(device)
                with torch.no_grad():
                    _ = net(warmup_batch)

                for w in args.widths:
                    dpath = str(data_root / f"w{w}_{args.init_type}")
                    spath = str(data_root / f"w{w}_{args.init_type}" / "mnist_sizegen_splits.json")
                    layout = [2, w, w, 1]

                    loader = build_test_loader(
                        conf_data, layout, dpath, spath, direction,
                        equiv_on_hidden, get_first_layer_mask, fanin_rescale,
                        args.batch_size
                    )

                    emb, lbl = extract_embeddings(net, loader, device)
                    embeddings_by_width[w] = emb
                    labels_by_width[w] = lbl

                    if "trained" in cond_name:
                        cache_file = output_dir / "embeddings" / f"emb_{cond_name}_w{w}.npz"
                    else:
                        cache_file = output_dir / "embeddings" / f"emb_{cond_name}_seed{seed}_w{w}.npz"
                    np.savez_compressed(cache_file, embeddings=emb, labels=lbl)

                del net
                torch.cuda.empty_cache()

            # Compute distances
            swd_mat, mmd_mat = compute_pairwise_distances(
                embeddings_by_width, args.widths, n_projections=args.n_projections
            )
            swd_matrices.append(swd_mat)
            mmd_matrices.append(mmd_mat)

            # Save per-seed distance matrices
            if "trained" in cond_name:
                np.savetxt(output_dir / "distances" / f"swd_{cond_name}.csv", swd_mat, delimiter=",")
                np.savetxt(output_dir / "distances" / f"mmd_{cond_name}.csv", mmd_mat, delimiter=",")
            else:
                np.savetxt(output_dir / "distances" / f"swd_{cond_name}_seed{seed}.csv", swd_mat, delimiter=",")
                np.savetxt(output_dir / "distances" / f"mmd_{cond_name}_seed{seed}.csv", mmd_mat, delimiter=",")

        # Aggregate across seeds
        swd_all = np.array(swd_matrices)
        mmd_all = np.array(mmd_matrices)
        np.savetxt(output_dir / "distances" / f"swd_{cond_name}_mean.csv", swd_all.mean(0), delimiter=",")
        np.savetxt(output_dir / "distances" / f"swd_{cond_name}_std.csv", swd_all.std(0), delimiter=",")
        np.savetxt(output_dir / "distances" / f"mmd_{cond_name}_mean.csv", mmd_all.mean(0), delimiter=",")
        np.savetxt(output_dir / "distances" / f"mmd_{cond_name}_std.csv", mmd_all.std(0), delimiter=",")

        all_results[cond_name] = {"matrices": swd_matrices, "mmd_matrices": mmd_matrices}

        # Heatmap for mean distance
        plot_heatmap(swd_all.mean(0), args.widths, f"SWD: {cond_name}",
                     plot_dir / f"heatmap_swd_{cond_name}_mean.pdf")

    # --- Convergence overlay plots ---
    print("\nGenerating convergence plots...")
    plot_convergence_curves(all_results, args.widths, output_dir, metric="swd")

    mmd_results = {k: {"matrices": v["mmd_matrices"]} for k, v in all_results.items()}
    plot_convergence_curves(mmd_results, args.widths, output_dir, metric="mmd")

    # --- 2D point cloud visualization ---
    print("\nGenerating 2D point cloud visualizations...")
    for fanin_rescale, tag in [(True, "dupequiv"), (False, "baseline")]:
        all_seeds_points = []
        for seed in range(min(args.n_random_seeds, 9)):
            conf = yaml.safe_load(yaml.dump(base_conf))
            conf = overwrite_conf(conf, {})
            conf = resolve_data_paths(conf)
            if fanin_rescale:
                conf["scalegmn_args"]["gnn_args"]["msg_equiv_on_hidden"] = True
            conf["scalegmn_args"]["layer_layout"] = conf["data"]["train_layer_layouts"][0]

            net = build_model_2d(conf, device, fanin_rescale, seed=seed + 100)

            direction = conf["scalegmn_args"]["direction"]
            equiv_on_hidden = mask_hidden(conf)
            get_first_layer_mask = mask_input(conf)
            conf_data = conf["data"]

            # Warmup
            warmup_loader = build_test_loader(
                conf_data, conf["data"]["train_layer_layouts"][0],
                conf["data"]["train_dataset_paths"][0],
                conf["data"]["train_split_paths"][0],
                direction, equiv_on_hidden, get_first_layer_mask,
                fanin_rescale, batch_size=4
            )
            warmup_batch = next(iter(warmup_loader)).to(device)
            with torch.no_grad():
                _ = net(warmup_batch)

            points_by_width = {}
            labels_by_width = {}
            for w in args.widths:
                cache_file = output_dir / "embeddings_2d" / f"emb2d_{tag}_seed{seed}_w{w}.npz"
                if cache_file.exists() and not args.recompute:
                    data = np.load(cache_file)
                    points_by_width[w] = data["points"]
                    labels_by_width[w] = data["labels"]
                else:
                    dpath = str(data_root / f"w{w}_{args.init_type}")
                    spath = str(data_root / f"w{w}_{args.init_type}" / "mnist_sizegen_splits.json")
                    layout = [2, w, w, 1]
                    loader = build_test_loader(
                        conf_data, layout, dpath, spath, direction,
                        equiv_on_hidden, get_first_layer_mask, fanin_rescale,
                        args.batch_size
                    )
                    pts, lbl = extract_2d_outputs(net, loader, device)
                    points_by_width[w] = pts
                    labels_by_width[w] = lbl
                    np.savez_compressed(cache_file, points=pts, labels=lbl)

            all_seeds_points.append(points_by_width)
            del net
            torch.cuda.empty_cache()

        if all_seeds_points:
            plot_2d_grid(all_seeds_points, args.widths,
                         plot_dir / f"pointcloud_2d_{tag}_grid.pdf")
            plot_2d_pointclouds(
                all_seeds_points[0], labels_by_width, args.widths,
                plot_dir / f"pointcloud_2d_{tag}_seed0.pdf",
                title=f"2D output ({tag}, seed 0)"
            )

    # --- Per-class analysis (dup-equiv, seed 0) ---
    print("\nPer-class analysis...")
    cond_key = "random-dupequiv"
    if cond_key in all_results:
        seed0_emb = {}
        seed0_lbl = {}
        for w in args.widths:
            cache_file = output_dir / "embeddings" / f"emb_{cond_key}_seed0_w{w}.npz"
            if cache_file.exists():
                data = np.load(cache_file)
                seed0_emb[w] = data["embeddings"]
                seed0_lbl[w] = data["labels"]

        if seed0_emb:
            perclass_swd = {}
            for cls in range(10):
                cls_emb = {w: seed0_emb[w][seed0_lbl[w] == cls] for w in args.widths}
                # Only compute if all widths have samples for this class
                if all(len(cls_emb[w]) > 0 for w in args.widths):
                    swd_mat, _ = compute_pairwise_distances(cls_emb, args.widths, n_projections=args.n_projections)
                    perclass_swd[cls] = swd_mat

            if perclass_swd:
                plot_perclass_summary(perclass_swd, args.widths,
                                      plot_dir / "perclass_swd_summary.pdf")

    # --- PCA/t-SNE on trained embeddings ---
    print("\nPCA visualization...")
    for cond_key in ["trained-dupequiv", "trained-baseline"]:
        emb_all = []
        width_ids = []
        for w in args.widths:
            cache_file = output_dir / "embeddings" / f"emb_{cond_key}_w{w}.npz"
            if cache_file.exists():
                data = np.load(cache_file)
                emb_all.append(data["embeddings"])
                width_ids.extend([w] * len(data["embeddings"]))

        if emb_all:
            from sklearn.decomposition import PCA
            emb_concat = np.concatenate(emb_all, axis=0)
            pca = PCA(n_components=2)
            emb_2d = pca.fit_transform(emb_concat)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            cmap = plt.cm.get_cmap("tab10", len(args.widths))
            width_ids = np.array(width_ids)
            for i, w in enumerate(args.widths):
                mask = width_ids == w
                ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[cmap(i)], s=3,
                           alpha=0.3, label=f"w{w}")
            ax.legend(markerscale=5)
            ax.set_title(f"PCA: {cond_key}")
            plt.tight_layout()
            plt.savefig(plot_dir / f"pca_{cond_key}.pdf", dpi=150)
            plt.close()

    # --- Monotonicity check ---
    print("\n" + "=" * 60)
    print("MONOTONICITY CHECK (distance to largest width)")
    print("=" * 60)
    ref_idx = len(args.widths) - 1
    for cond_name, data in all_results.items():
        matrices = data["matrices"]
        dists_to_ref = np.array([m[:, ref_idx] for m in matrices])  # [K, n_widths]
        mean_dists = dists_to_ref.mean(axis=0)
        is_monotone = all(mean_dists[i] >= mean_dists[i + 1] for i in range(ref_idx - 1))
        mono_str = "YES" if is_monotone else "NO"
        print(f"\n{cond_name} (monotone={mono_str}):")
        for i, w in enumerate(args.widths[:-1]):
            print(f"  w{w} → w{args.widths[-1]}: SWD = {mean_dists[i]:.4f} "
                  f"(±{dists_to_ref.std(axis=0)[i]:.4f})")

    print(f"\nPlots saved to: {plot_dir}")
    print(f"Embeddings/distances saved to: {output_dir}")


if __name__ == "__main__":
    main()
