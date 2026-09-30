"""
Duplication equivalence of the GMN on **CNN graphs** (the CNN analog of
``scripts/check_duplication_equiv.py``).

Setup
-----
Base network: the zoo CNN ``[1] -> conv3x3/s2 x3 -> [c] -> GAP -> dense -> [10]``.
Widened network: channel multipliers ``k_1 = k_2 = k_3 = k`` with the uniform
(pure-duplication) factor ``P^(l) = 1 1^T / k_{l-1}``, which
``scripts/check_cnn_widening_equiv.py`` shows represents the same function.

Why the three modifications carry over unchanged
------------------------------------------------
1. **Fan-in rescaling.** After ``_transform_weights_biases`` a layer's weight is
   ``[n_in, n_out, k_h*k_w]`` and the graph puts one edge per *input channel*,
   carrying the whole (zero-padded) R^9 kernel.  Rescaling by ``n_in`` therefore
   makes the widened edge feature equal to the base one entry-by-entry:

       n_in * k_in * W[i,j,s] * P[b,a] = n_in * k_in * W[i,j,s] / k_in
                                       = n_in * W[i,j,s].

   The factor is ``n_in``, not ``n_in * k_h * k_w`` — mean aggregation averages
   over incoming *edges*, and there are ``n_in`` of them.
2. **Mean aggregation.** All ``k_in`` messages into a widened node are then
   identical, so their mean equals the base node's aggregate.  Node features are
   biases, and ``b_up[(i,b)] = b[i]``, so by induction over layers every copy of a
   base channel gets the base channel's representation.
3. **Readout.** Two work, and they are checked separately:
     * ``layerwise_mean`` — per-layer mean pooling over hidden nodes
       (:mod:`src.models.layerwise_mean_readout`), as on the INR side;
     * ``last_layer`` — reads only the output-layer nodes
       (:mod:`src.models.last_layer_readout`).  Those are never duplicated
       (``k_L = 1``), so this is duplication-invariant *by construction* and needs
       only fan-in + mean.
   The upstream sum-pooling readout (``PermScaleInvariantReadout`` for
   ``symmetry: scale``, ``DeepSet`` for ``permutation``) is included as the
   negative control: sum over ``k_l`` copies scales with ``k_l``.

Expected outcome, per symmetry and per readout: PASS (< tol) only with
fan-in + mean; no-fan-in and sum aggregation both FAIL, and so does the
sum-pooling readout even when the node representations are correct.  Tested in
both ``forward`` and ``bidirectional`` direction (``--forward-only`` drops the
latter, for models with no bidirectional variant — mpsgmn_*), the same split as
``scripts/check_duplication_equiv.py`` on MLP graphs.  Uniform duplication
duplicates every edge feature identically regardless of direction, so the
bidirectional aggregate needs no ``install_bidir_reciprocal`` fix to respect
it — that fix is what makes *scale equivariance* hold bidirectionally
(``check_scale_equiv_cnn.py``), a different property from the one here.

Also reported (informational): whether ``cnn_to_tg_data``'s
``bw_edge_attr = reciprocal(edge_attr)`` is finite.  It is not — every kernel
zero-padded channel, and every zero weight, reciprocates to ``inf``.  That field
is nonetheless **dead**: ``ScaleGMN_GNN_bidir.forward`` initialises
``bw_edge_attr = batch.edge_attr`` and never reads ``batch.bw_edge_attr`` (it uses
only ``batch.bw_edge_index``), so a bidirectional CNN model does run and produces
finite outputs — this checker's own bidirectional conditions below are unaffected
by the ``inf``. The CNN accuracy-prediction task's own configs are still
``direction: forward`` by default (a design choice unrelated to this checker's
scope; `direction` is a CLI override at train/eval time — see `README.md`).

Usage
-----
    python scripts/check_duplication_equiv_cnn.py
    python scripts/check_duplication_equiv_cnn.py --symmetry scale --readout last_layer
    python scripts/check_duplication_equiv_cnn.py --width 16 --k 3 --n-nets 5
    python scripts/check_duplication_equiv_cnn.py --forward-only --template-conf ...mpsgmn...
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch_geometric
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
SCALEGMN_ROOT = PROJECT_ROOT / "src" / "scalegmn"

os.chdir(SCALEGMN_ROOT)
sys.path.insert(0, str(SCALEGMN_ROOT))

from src.scalegmn.models import ScaleGMN                                   # noqa: E402
from src.utils.helpers import overwrite_conf                               # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from data.cnn_zoo_dataset import build_cnn_graph_data                      # noqa: E402
from data.zoo_cnn import (                                                 # noqa: E402
    ZooCNN, layer_layout, state_dict_to_wb, widen_wb,
)
from models.last_layer_readout import install_last_layer_readout           # noqa: E402
from models.layerwise_mean_readout import install_layerwise_mean_readout    # noqa: E402
from models.matrix_product_scale_layer import matrix_product_context        # noqa: E402

#: Architecture template only — everything the ablation varies (aggregator, fan-in,
#: readout) is overridden in build_conf().  Overridable with --template-conf, which is
#: how the matrix-product variants (mpgmn_*, mpsgmn_*) are checked.
TEMPLATE_CONF = PROJECT_ROOT / "configs" / "cifar10_predgen" / "scalegmn_predgen_sp.yml"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_conf(base_layout, aggregator, symmetry, direction="forward"):
    """Template config overridden for a single equivalence check."""
    with open(TEMPLATE_CONF) as f:
        conf = yaml.safe_load(f)
    conf = overwrite_conf(conf, {})
    args = conf["scalegmn_args"]
    args["d_hid"] = 32                     # small, to keep the check fast
    args["num_layers"] = 2
    args["layer_layout"] = base_layout
    args["direction"] = direction
    args["symmetry"] = symmetry
    args["readout_range"] = "full_graph"   # our readouts install into this slot
    args["gnn_args"]["aggregator"] = aggregator
    if symmetry == "permutation":
        args["positional_encodings"]["sum_pos_enc"] = True
        args["positional_encodings"]["equiv_on_hidden"] = False
        args["gnn_args"]["msg_equiv_on_hidden"] = False
        args["gnn_args"]["upd_equiv_on_hidden"] = False
        args["gnn_args"]["sign_symmetrization"] = False
    return overwrite_conf(conf, {})        # re-propagate d_hid


def build_model(base_layout, aggregator, symmetry, readout, direction="forward"):
    conf = build_conf(base_layout, aggregator, symmetry, direction)
    # No-op unless the template config selects a matrix-product MSG function.
    with matrix_product_context(conf["scalegmn_args"]):
        model = ScaleGMN(conf["scalegmn_args"])
    if readout == "layerwise_mean":
        install_layerwise_mean_readout(model, conf["scalegmn_args"])
    elif readout == "last_layer":
        install_last_layer_readout(model, conf["scalegmn_args"])
    elif readout != "sum":
        raise ValueError(readout)
    model.eval()
    return model


def copy_backbone(src_model, dst_model):
    """Copy every shape-compatible parameter (i.e. all but the readout)."""
    src_sd, dst_sd = src_model.state_dict(), dst_model.state_dict()
    for key in dst_sd:
        if key in src_sd and src_sd[key].shape == dst_sd[key].shape:
            dst_sd[key] = src_sd[key]
    dst_model.load_state_dict(dst_sd)
    dst_model.eval()


# ---------------------------------------------------------------------------
# Forward passes
# ---------------------------------------------------------------------------

def node_features(model, data):
    """Node representations before the readout (equivariant mode)."""
    batch = torch_geometric.data.Batch.from_data_list([data])
    model.gnn.equivariant = True
    with torch.no_grad():
        x, _ = model(batch)
    model.gnn.equivariant = False
    return x


def graph_output(model, data):
    batch = torch_geometric.data.Batch.from_data_list([data])
    with torch.no_grad():
        return model(batch)


def compare_node_feats(x_base, x_wide, base_layout, multipliers):
    """Max abs error between each base node and its ``k_l`` widened copies."""
    max_err = 0.0
    off_base = off_wide = 0
    for n_l, k_l in zip(base_layout, multipliers):
        for j in range(n_l):
            base = x_base[off_base + j]
            wide = x_wide[off_wide + j * k_l: off_wide + (j + 1) * k_l]
            max_err = max(max_err, (wide - base).abs().max().item())
        off_base += n_l
        off_wide += n_l * k_l
    return max_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symmetry", choices=["scale", "permutation", "both"], default="both")
    p.add_argument("--readout", choices=["layerwise_mean", "last_layer", "both"],
                   default="both")
    p.add_argument("--width", type=int, default=16, help="Base channel count")
    p.add_argument("--k", type=int, nargs="+", default=[2, 3],
                   help="Width multipliers to test")
    p.add_argument("--n-nets", type=int, default=3, help="Random CNNs per k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--template-conf", type=Path, default=TEMPLATE_CONF,
                   help="Architecture template config. Pass a matrix-product config "
                        "(mpgmn_*/mpsgmn_*) to check that MSG function instead.")
    p.add_argument("--forward-only", action="store_true",
                   help="Skip the bidirectional conditions. Use for mpsgmn_* (the "
                        "conv matrix-product ScaleGMN is forward-only in every "
                        "experiment). The forward numbers are unchanged by this flag.")
    return p.parse_args()


def main():
    global TEMPLATE_CONF
    args = parse_args()
    TEMPLATE_CONF = args.template_conf if args.template_conf.is_absolute() \
        else (PROJECT_ROOT / args.template_conf)
    torch.manual_seed(args.seed)

    symmetries = ["scale", "permutation"] if args.symmetry == "both" else [args.symmetry]
    readouts = ["layerwise_mean", "last_layer"] if args.readout == "both" else [args.readout]

    base_layout = layer_layout(args.width)
    print("Duplication equivalence on CNN graphs")
    msg_fn = build_conf(base_layout, "mean", symmetries[0])["scalegmn_args"]["gnn_args"] \
        .get("message_fn_type")
    print(f"template={TEMPLATE_CONF.name}  message_fn_type={msg_fn}")
    directions = ["forward"] if args.forward_only else ["forward", "bidirectional"]
    dir_tag = {"forward": "fw", "bidirectional": "bd"}

    print(f"base layout={base_layout}  k={args.k}  nets/k={args.n_nets}  "
          f"tol={args.tol:.0e}  directions={directions}")
    print()

    # --- informational: the bidirectional reciprocal trap -------------------
    net = ZooCNN(width=args.width)
    w, b = state_dict_to_wb(net.export_state_dict())
    bd_probe = build_cnn_graph_data(w, b, direction="bidirectional", fanin_rescale=True)
    n_inf = torch.isinf(bd_probe.bw_edge_attr).sum().item()
    frac = n_inf / bd_probe.bw_edge_attr.numel()
    print(f"[info] direction=bidirectional: bw_edge_attr = reciprocal(edge_attr) has "
          f"{n_inf}/{bd_probe.bw_edge_attr.numel()} ({frac:.1%}) non-finite entries "
          f"(zero-padded kernel channels).")
    print("       That field is dead: ScaleGMN_GNN_bidir.forward sets bw_edge_attr = "
          "batch.edge_attr and\n       reads only batch.bw_edge_index, so the "
          "bidirectional conditions below run and\n       produce finite outputs "
          "regardless of the infs.")
    print()

    # --- graph builders ----------------------------------------------------
    def graphs(weights, biases, fanin, direction):
        return build_cnn_graph_data(weights, biases, direction=direction,
                                    fanin_rescale=fanin, score=0.0,
                                    node_pos_embed=True, edge_pos_embed=False,
                                    equiv_on_hidden=True, get_first_layer_mask=False)

    all_ok = True
    for sym in symmetries:
        # One backbone per (symmetry, direction), shared across aggregators and
        # readouts within that direction so the aggregator/readout is the only
        # variable. Forward and bidirectional models are independent (the
        # bidirectional GNN layer carries its own backward-direction parameters),
        # each checked for its own duplication equivalence — no reciprocal fix
        # needed, see module docstring. Both are built even under --forward-only,
        # where the bidirectional ones go unused: they still consume RNG, so
        # building them unconditionally keeps the reported forward numbers
        # identical in both modes (mirrors check_duplication_equiv.py).
        m_mean, m_sum, ro_models = {}, {}, {}
        for direction in dir_tag:
            dt = dir_tag[direction]
            m_mean[dt] = build_model(base_layout, "mean", sym, "sum", direction)
            m_sum[dt] = build_model(base_layout, "add", sym, "sum", direction)
            copy_backbone(m_mean[dt], m_sum[dt])
            for ro in readouts:
                for aggr, tag in (("mean", "mean"), ("add", "sum")):
                    m = build_model(base_layout, aggr, sym, ro, direction)
                    copy_backbone(m_mean[dt], m)
                    ro_models[(dt, ro, tag)] = m

        keys = []
        for direction in directions:
            dt = dir_tag[direction]
            keys += [f"{dt}/node/fanin+mean", f"{dt}/node/nofanin+mean",
                     f"{dt}/node/fanin+sum", f"{dt}/graph/sum-ro/fanin+mean"]
            for ro in readouts:
                keys += [f"{dt}/graph/{ro}/fanin+mean", f"{dt}/graph/{ro}/fanin+sum",
                         f"{dt}/graph/{ro}/nofanin+mean"]
        errs = {k: 0.0 for k in keys}

        for k in args.k:
            for _ in range(args.n_nets):
                net = ZooCNN(width=args.width)
                weights, biases = state_dict_to_wb(net.export_state_dict())
                mults = [1] + [k] * (len(weights) - 1) + [1]
                wide_w, wide_b = widen_wb(weights, biases, mults, "uniform")

                def upd(key, err):
                    errs[key] = max(errs[key], err)

                for direction in directions:
                    dt = dir_tag[direction]
                    g_base_fi = graphs(weights, biases, True, direction)
                    g_wide_fi = graphs(wide_w, wide_b, True, direction)
                    g_base_no = graphs(weights, biases, False, direction)
                    g_wide_no = graphs(wide_w, wide_b, False, direction)

                    # node level
                    upd(f"{dt}/node/fanin+mean", compare_node_feats(
                        node_features(m_mean[dt], g_base_fi), node_features(m_mean[dt], g_wide_fi),
                        base_layout, mults))
                    upd(f"{dt}/node/nofanin+mean", compare_node_feats(
                        node_features(m_mean[dt], g_base_no), node_features(m_mean[dt], g_wide_no),
                        base_layout, mults))
                    upd(f"{dt}/node/fanin+sum", compare_node_feats(
                        node_features(m_sum[dt], g_base_fi), node_features(m_sum[dt], g_wide_fi),
                        base_layout, mults))

                    # graph level — upstream sum-pooling readout (negative control)
                    upd(f"{dt}/graph/sum-ro/fanin+mean",
                        (graph_output(m_mean[dt], g_base_fi)
                         - graph_output(m_mean[dt], g_wide_fi)).abs().max().item())

                    # graph level — our width-agnostic readouts
                    for ro in readouts:
                        mm, ms = ro_models[(dt, ro, "mean")], ro_models[(dt, ro, "sum")]
                        upd(f"{dt}/graph/{ro}/fanin+mean",
                            (graph_output(mm, g_base_fi)
                             - graph_output(mm, g_wide_fi)).abs().max().item())
                        upd(f"{dt}/graph/{ro}/fanin+sum",
                            (graph_output(ms, g_base_fi)
                             - graph_output(ms, g_wide_fi)).abs().max().item())
                        upd(f"{dt}/graph/{ro}/nofanin+mean",
                            (graph_output(mm, g_base_no)
                             - graph_output(mm, g_wide_no)).abs().max().item())

        expect = {}
        for direction in directions:
            dt = dir_tag[direction]
            expect.update({f"{dt}/node/fanin+mean": True, f"{dt}/node/nofanin+mean": False,
                           f"{dt}/node/fanin+sum": False, f"{dt}/graph/sum-ro/fanin+mean": False})
            for ro in readouts:
                expect[f"{dt}/graph/{ro}/fanin+mean"] = True
                expect[f"{dt}/graph/{ro}/fanin+sum"] = False
                expect[f"{dt}/graph/{ro}/nofanin+mean"] = False

        print(f"symmetry = {sym}")
        for key in keys:
            max_err = errs[key]
            passed = max_err < args.tol
            correct = passed == expect[key]
            all_ok = all_ok and correct
            print(f"  {'OK ' if correct else 'BAD'} [{'PASS' if passed else 'FAIL'}]  "
                  f"{key:<34} (should {'PASS' if expect[key] else 'FAIL'})  "
                  f"max_err={max_err:.2e}")
        print()

    if all_ok:
        print("[OK] All conditions behaved as expected.")
    else:
        print("[FAIL] Some conditions did not behave as expected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
