"""
Batch INR fitting utilities.

Two fitters are provided:
  - BatchINRFitter: standard parameterization (SP), ReLU hidden layers.
  - MuBatchINRFitter: muP (maximal update parameterization) ReLU variant.

Weight storage convention: BOTH fitters store weights as [out_dim, in_dim] in state
dicts. ScaleGMN's BaseDataset calls permute(1, 0) on load, converting to [in, out].
The graph construction in batch_to_graphs() requires weights[0].shape[0] == n_input,
which holds only when stored weights are [out, in].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchINRFitter(nn.Module):
    """Fit B independent ReLU INRs in parallel using batched matmul.

    Architecture: dims[0] -> dims[1] -> ... -> dims[-1]
    Activations: ReLU for all hidden layers, Sigmoid for the output layer.

    Weights are stored as [B, in_dim, out_dim] for use with torch.bmm.
    """

    def __init__(self, batch_size: int, dims: list[int] = [2, 32, 32, 1]):
        super().__init__()
        self.dims = dims
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(len(dims) - 1):
            std = (2.0 / dims[i]) ** 0.5
            w = torch.randn(batch_size, dims[i], dims[i + 1]) * std
            b = torch.zeros(batch_size, 1, dims[i + 1])
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(b))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [P, 2] coordinate grid
        Returns:
            [B, P] predicted pixel intensities
        """
        B = self.weights[0].size(0)
        x = coords.unsqueeze(0).expand(B, -1, -1)  # [B, P, 2]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.bmm(x, w) + b
            if i < len(self.weights) - 1:
                x = F.relu(x)
            else:
                x = torch.sigmoid(x)

        return x.squeeze(-1)  # [B, P]

    def get_state_dicts(self) -> list[dict[str, torch.Tensor]]:
        """Return list of B state dicts compatible with ScaleGMN's BaseDataset.

        Keys: 'layers.{i}.weight' with shape [out_dim, in_dim]
              'layers.{i}.bias'   with shape [out_dim]

        Weights are stored as [out, in] so that ScaleGMN's permute(1, 0) converts
        them to [in, out], and weights[0].shape[0] == n_input_nodes as required by
        batch_to_graphs().
        """
        B = self.weights[0].size(0)
        state_dicts = []
        for b_idx in range(B):
            sd = {}
            for i, (w, bias) in enumerate(zip(self.weights, self.biases)):
                # w[b_idx]: [in, out] -> .t() -> [out, in]
                sd[f"layers.{i}.weight"] = w[b_idx].detach().cpu().t()  # [out, in]
                sd[f"layers.{i}.bias"] = bias[b_idx, 0].detach().cpu()  # [out]
            state_dicts.append(sd)
        return state_dicts


class MuBatchINRFitter(nn.Module):
    """Fit B independent ReLU INRs in parallel using batched matmul with muP scaling.

    Implements maximal update parameterization (muP) for a [2, w, w, 1] INR:
      - Kaiming He init; output layer scaled by sqrt(width_mult) to compensate
        for the 1/wm division in forward, keeping init output magnitude O(1)
      - Hidden→hidden weight LR scaled by base_width/width (ninf=2 in muP)
      - Output layer forward divides input by width_mult = width/base_width
      - Stored readout weights are effective: W_eff = W / width_mult

    Architecture: dims[0] -> dims[1] -> ... -> dims[-1]
    Activations: ReLU for hidden layers, Sigmoid for output.
    """

    def __init__(
        self,
        batch_size: int,
        dims: list[int] = [2, 32, 32, 1],
        base_width: int = 32,
    ):
        super().__init__()
        self.dims = dims
        self.batch_size = batch_size
        self.base_width = base_width
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            std = (2.0 / in_d) ** 0.5
            w = torch.randn(batch_size, in_d, out_d) * std
            b = torch.zeros(batch_size, 1, out_d)

            if i == len(dims) - 2:  # output layer
                w.data *= (in_d / base_width) ** 0.5

            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(b))

    def get_mup_optimizer(self, lr: float):
        """Return Adam optimizer with muP per-layer LR scaling.

        muP rule: weight matrices with ninf=2 (both dims infinite) get LR /= width_mult.
        For [2, w, w, 1]: only layers.1.weight (hidden→hidden) has ninf=2.
        All other params (ninf<=1) keep the base LR.
        """
        param_groups = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            in_d, out_d = self.dims[i], self.dims[i + 1]
            if i > 0 and i < len(self.weights) - 1:  # hidden layer weights
                wm = in_d / self.base_width
                w_lr = lr / wm
            else:
                w_lr = lr
            param_groups.append({"params": [w], "lr": w_lr})
            param_groups.append({"params": [b], "lr": lr})
        return torch.optim.Adam(param_groups)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [P, 2] coordinate grid
        Returns:
            [B, P] predicted pixel intensities
        """
        B = self.weights[0].size(0)
        x = coords.unsqueeze(0).expand(B, -1, -1)  # [B, P, 2]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if i == len(self.weights) - 1:
                # MuReadout: divide input by width_mult before linear
                wm = self.dims[i] / self.base_width
                x = torch.bmm(x / wm, w) + b
                x = torch.sigmoid(x)
            else:
                x = torch.bmm(x, w) + b
                x = F.relu(x)

        return x.squeeze(-1)  # [B, P]

    def get_state_dicts(self) -> list[dict[str, torch.Tensor]]:
        """Return list of B state dicts compatible with ScaleGMN's BaseDataset.

        Keys: 'layers.{i}.weight' [out, in], 'layers.{i}.bias' [out]

        Stores effective readout weights (W / width_mult) so downstream
        evaluation uses a plain forward pass without muP scaling.
        """
        B = self.weights[0].size(0)
        state_dicts = []
        for b_idx in range(B):
            sd = {}
            for i, (w, bias) in enumerate(zip(self.weights, self.biases)):
                # w[b_idx]: [in, out] -> .t() -> [out, in]
                weight = w[b_idx].detach().cpu().t()  # [out, in]
                b_val = bias[b_idx, 0].detach().cpu()  # [out]
                if i == len(self.weights) - 1:
                    # Store effective weight: W_eff = W / width_mult
                    wm = self.dims[i] / self.base_width
                    weight = weight / wm
                sd[f"layers.{i}.weight"] = weight
                sd[f"layers.{i}.bias"] = b_val
            state_dicts.append(sd)
        return state_dicts
