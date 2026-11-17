"""Tabular feature encoders."""

import torch
from torch import nn


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
