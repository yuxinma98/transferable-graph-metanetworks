"""Datasets for the CNN-zoo accuracy-prediction task.

Two classes, mirroring the INR side of the repo:

* :class:`CNNZooDataset` — reads a ``{"path": [...], "score": [...]}`` split JSON of
  ``.pth`` state dicts and builds one PyG graph per CNN.  It exists because
  upstream's :class:`CNNDataset` (``src/scalegmn/src/data/cifar10_dataset.py``)
  is broken on that path: its ``__getitem__`` calls
  ``cnn_to_tg_data(weights, biases, conv_mask, fmap_size=...)`` while the
  signature is ``(weights, biases, conv_mask, direction, ...)``, so ``fmap_size``
  binds nothing and ``direction`` is missing → ``TypeError``.  Rather than patch
  the git subtree we re-implement ``__getitem__`` here, modelled on
  ``NFNZooDataset.__getitem__`` (which does pass ``direction``), and reuse
  upstream's ``_transform_weights_biases`` / ``cnn_to_tg_data`` verbatim.

* :class:`FanInCNNZooDataset` — the fan-in-rescaled variant, exactly analogous to
  :class:`src.data.duplication_equiv_dataset.FanInLabeledINRDataset`.  Each weight
  ``W^(l)``, after being reshaped to ``[n_in, n_out, k_h*k_w]``, is multiplied by
  its number of input **channel nodes** ``n_in = w.shape[0]``.  Not
  ``n_in * k_h * k_w``: mean aggregation averages over incoming *edges*, and the
  graph has one edge per input channel (its feature is the whole R^9 kernel).

Graph conventions (all inherited from upstream ``cnn_to_graph``):
  * nodes = channels, plus one node per dense output unit; input nodes carry
    feature 0, every other node carries its bias;
  * edges = layer-to-layer only, feature = the 3x3 kernel zero-padded and
    flattened to R^9 (dense-layer edges put their scalar in channel 0);
  * node layout ``[in_channels | conv1 | conv2 | conv3 | dense-out]``, so
    ``node2type`` from ``get_node_types`` gives each input/output node its own
    type and each hidden layer one shared type.  ``num_param_types`` is
    ``1 + 3 + 10 = 14`` at every width, so the positional encodings transfer
    across widths unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_SCALEGMN_ROOT = Path(__file__).resolve().parent.parent / "scalegmn"
if str(_SCALEGMN_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCALEGMN_ROOT))

from src.data.cifar10_dataset import CNNDataset, cnn_to_tg_data  # noqa: E402
from src.data.data_utils import get_edge_types, get_node_types    # noqa: E402


def _mark_hidden_nodes(layer_layout) -> torch.Tensor:
    return torch.tensor(
        [False] * layer_layout[0]
        + [True] * sum(layer_layout[1:-1])
        + [False] * layer_layout[-1]
    ).unsqueeze(-1)


def _mark_input_nodes(layer_layout) -> torch.Tensor:
    return torch.tensor(
        [True] * layer_layout[0] + [False] * sum(layer_layout[1:])
    ).unsqueeze(-1)


def build_cnn_graph_data(
    weights,
    biases,
    *,
    direction="forward",
    fanin_rescale=False,
    score=0.0,
    max_kernel_size=(3, 3),
    node_pos_embed=True,
    edge_pos_embed=False,
    equiv_on_hidden=True,
    get_first_layer_mask=False,
    activation_function="relu",
):
    """Build one PyG ``Data`` object from raw CNN weight / bias tensors.

    Shared by :class:`CNNZooDataset` and the verification scripts so that both go
    through exactly the same graph construction.

    Args:
        weights: ``[out, in, kh, kw]`` per conv layer, then ``[out, in]`` for the head.
        biases: ``[out]`` per layer.
        fanin_rescale: multiply each transformed weight by its fan-in ``n_in``
            (required, together with mean aggregation, for duplication equivalence).
    """
    conv_mask = [1 if w.ndim == 4 else 0 for w in weights]
    layer_layout = [weights[0].shape[1]] + [b.shape[0] for b in biases]

    tw = [CNNDataset._transform_weights_biases(w, max_kernel_size, linear_as_conv=False)
          for w in weights]
    tb = [CNNDataset._transform_weights_biases(b, max_kernel_size, linear_as_conv=False)
          for b in biases]
    if fanin_rescale:
        # w: [n_in, n_out, k_h*k_w]; w.shape[0] == n_in == number of incoming edges
        tw = [w * w.shape[0] for w in tw]

    return cnn_to_tg_data(
        tuple(tw),
        tuple(tb),
        conv_mask,
        direction,
        fmap_size=1,
        y=float(score),
        layer_layout=layer_layout,
        node2type=get_node_types(layer_layout) if node_pos_embed else None,
        edge2type=get_edge_types(layer_layout) if edge_pos_embed else None,
        mask_hidden=_mark_hidden_nodes(layer_layout) if equiv_on_hidden else None,
        mask_first_layer=_mark_input_nodes(layer_layout) if get_first_layer_mask else None,
        sign_mask=activation_function == "tanh",
    )


class CNNZooDataset(torch.utils.data.Dataset):
    """A ``{"path", "score"}`` split of ``.pth`` CNN state dicts, as PyG graphs.

    All CNNs in one instance must share an architecture (one width per instance),
    which is how the size-generalization harness uses it: one dataset per test
    width.  ``layer_layout`` is taken from the config when given and otherwise
    read off the first item.
    """

    fanin_rescale = False

    def __init__(
        self,
        dataset=None,
        dataset_path=".",
        split_path=None,
        split="train",
        debug=False,
        node_pos_embed=False,
        edge_pos_embed=False,
        equiv_on_hidden=False,
        get_first_layer_mask=False,
        layer_layout=None,
        direction="forward",
        activation_function="relu",
        max_kernel_size=(3, 3),
        max_items=None,
        **unused,
    ):
        self.split = split
        self.dataset_path = Path(dataset_path)
        splits_path = Path(split_path)
        if not splits_path.is_absolute():
            splits_path = self.dataset_path / splits_path
        with open(splits_path) as f:
            self.records = json.load(f)[split]
        self.paths = [
            p if Path(p).is_absolute() else (self.dataset_path / p).as_posix()
            for p in self.records["path"]
        ]
        self.scores = [float(s) for s in self.records["score"]]
        if debug:
            self.paths, self.scores = self.paths[:16], self.scores[:16]
        if max_items is not None:
            self.paths, self.scores = self.paths[:max_items], self.scores[:max_items]

        self.node_pos_embed = node_pos_embed
        self.edge_pos_embed = edge_pos_embed
        self.equiv_on_hidden = equiv_on_hidden
        self.get_first_layer_mask = get_first_layer_mask
        self.direction = direction
        self.activation_function = activation_function
        self.max_kernel_size = tuple(max_kernel_size)
        self.layer_layout = list(layer_layout) if layer_layout is not None else None

    def __len__(self):
        return len(self.paths)

    def get_layer_layout(self):
        if self.layer_layout is None:
            self.layer_layout = list(self[0].layer_layout)
        return self.layer_layout

    def __getitem__(self, idx):
        state_dict = torch.load(self.paths[idx], map_location="cpu", weights_only=True)
        weights = [v for k, v in state_dict.items() if "weight" in k]
        biases = [v for k, v in state_dict.items() if "bias" in k]

        data = build_cnn_graph_data(
            weights,
            biases,
            direction=self.direction,
            fanin_rescale=self.fanin_rescale,
            score=self.scores[idx],
            max_kernel_size=self.max_kernel_size,
            node_pos_embed=self.node_pos_embed,
            edge_pos_embed=self.edge_pos_embed,
            equiv_on_hidden=self.equiv_on_hidden,
            get_first_layer_mask=self.get_first_layer_mask,
            activation_function=self.activation_function,
        )
        if self.layer_layout is not None:
            assert list(data.layer_layout) == self.layer_layout, (
                f"{self.paths[idx]} has layout {list(data.layer_layout)}, "
                f"expected {self.layer_layout}")
        return data


class FanInCNNZooDataset(CNNZooDataset):
    """:class:`CNNZooDataset` with fan-in edge rescaling.

    Use together with ``aggregator: mean`` for duplication-equivariant training,
    exactly as :class:`FanInLabeledINRDataset` is used on the INR side.
    """

    fanin_rescale = True
