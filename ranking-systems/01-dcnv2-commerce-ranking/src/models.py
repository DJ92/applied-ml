from __future__ import annotations

from typing import Any

import torch
from torch import nn


class EmbeddingBackbone(nn.Module):
    def __init__(self, cardinalities: list[int], dense_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(cardinality, embedding_dim) for cardinality in cardinalities])
        self.output_dim = dense_dim + embedding_dim * len(cardinalities)

    def forward(self, categorical: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        parts = []
        for index, embedding in enumerate(self.embeddings):
            parts.append(embedding(categorical[:, index]))
        parts.append(dense)
        return torch.cat(parts, dim=1)


class CrossLayer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        interaction = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x + x0 * interaction + self.bias


class LowRankCrossLayer(nn.Module):
    def __init__(self, input_dim: int, rank_dim: int) -> None:
        super().__init__()
        self.rank_down = nn.Linear(input_dim, rank_dim, bias=False)
        self.rank_up = nn.Linear(rank_dim, input_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        projected = self.rank_up(self.rank_down(x))
        return x + x0 * projected + self.bias


class LogisticRanker(nn.Module):
    def __init__(self, cardinalities: list[int], dense_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.backbone = EmbeddingBackbone(cardinalities, dense_dim, embedding_dim)
        self.head = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, categorical: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(categorical, dense)).squeeze(-1)


class MLPRanker(nn.Module):
    def __init__(self, cardinalities: list[int], dense_dim: int, embedding_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        self.backbone = EmbeddingBackbone(cardinalities, dense_dim, embedding_dim)
        layers: list[nn.Module] = []
        input_dim = self.backbone.output_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim, 1)

    def forward(self, categorical: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        hidden = self.mlp(self.backbone(categorical, dense))
        return self.head(hidden).squeeze(-1)


class DCNRanker(nn.Module):
    def __init__(self, cardinalities: list[int], dense_dim: int, embedding_dim: int, cross_layers: int) -> None:
        super().__init__()
        self.backbone = EmbeddingBackbone(cardinalities, dense_dim, embedding_dim)
        self.cross = nn.ModuleList([CrossLayer(self.backbone.output_dim) for _ in range(cross_layers)])
        self.head = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, categorical: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        base = self.backbone(categorical, dense)
        current = base
        for layer in self.cross:
            current = layer(base, current)
        return self.head(current).squeeze(-1)


class DCNV2Ranker(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        dense_dim: int,
        embedding_dim: int,
        cross_layers: int,
        low_rank_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = EmbeddingBackbone(cardinalities, dense_dim, embedding_dim)
        base_dim = self.backbone.output_dim
        self.cross = nn.ModuleList([LowRankCrossLayer(base_dim, low_rank_dim) for _ in range(cross_layers)])
        deep_layers: list[nn.Module] = []
        input_dim = base_dim
        for hidden_dim in hidden_dims:
            deep_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_dim
        self.deep = nn.Sequential(*deep_layers)
        self.head = nn.Linear(base_dim + input_dim, 1)

    def forward(self, categorical: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        base = self.backbone(categorical, dense)
        cross_out = base
        for layer in self.cross:
            cross_out = layer(base, cross_out)
        deep_out = self.deep(base)
        return self.head(torch.cat([cross_out, deep_out], dim=1)).squeeze(-1)


def build_model(name: str, cardinalities: list[int], dense_dim: int, config: dict[str, Any]) -> nn.Module:
    embedding_dim = int(config.get("embedding_dim", 16))
    hidden_dims = list(config.get("deep_dims", [64, 32]))
    dropout = float(config.get("dropout", 0.1))
    cross_layers = int(config.get("cross_layers", 2))
    low_rank_dim = int(config.get("low_rank_dim", 16))
    if name == "logistic":
        return LogisticRanker(cardinalities, dense_dim, embedding_dim)
    if name == "mlp":
        return MLPRanker(cardinalities, dense_dim, embedding_dim, hidden_dims, dropout)
    if name == "dcn":
        return DCNRanker(cardinalities, dense_dim, embedding_dim, cross_layers)
    if name == "dcnv2":
        return DCNV2Ranker(cardinalities, dense_dim, embedding_dim, cross_layers, low_rank_dim, hidden_dims, dropout)
    raise ValueError(f"Unsupported model: {name}")
