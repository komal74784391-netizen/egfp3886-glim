"""Gated latent interaction layer for tabular retrieval."""

import torch
from torch import nn


class GatedInteraction(nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        self.value = nn.Linear(dim * 2, dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([left, right], dim=-1)))
        value = self.value(torch.cat([left, right], dim=-1))
        return gate * value
