import torch


def make_coord_grid(H: int = 28, W: int = 28) -> torch.Tensor:
    """Return [H*W, 2] coordinate grid in [-1, 1]^2, ordered (x, y)."""
    ys = torch.linspace(-1, 1, H)
    xs = torch.linspace(-1, 1, W)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    return coords  # [H*W, 2]
