from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TowerBackbone(nn.Module):
    def __init__(self, cardinalities: list[int], dense_dim: int, embedding_dim: int, tower_dims: list[int], output_dim: int, dropout: float) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(cardinality, embedding_dim) for cardinality in cardinalities])
        input_dim = dense_dim + embedding_dim * len(cardinalities)
        layers: list[nn.Module] = []
        for hidden_dim in tower_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, categorical: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        parts = [embedding(categorical[:, index]) for index, embedding in enumerate(self.embeddings)]
        parts.append(dense)
        joined = torch.cat(parts, dim=1)
        output = self.network(joined)
        return torch.nn.functional.normalize(output, dim=1)


class MatrixFactorizationModel(nn.Module):
    def __init__(self, user_cardinality: int, item_cardinality: int, output_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(user_cardinality, output_dim)
        self.item_embedding = nn.Embedding(item_cardinality, output_dim)

    def user_vector(self, user_categorical: torch.Tensor, user_dense: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.user_embedding(user_categorical[:, 0]), dim=1)

    def item_vector(self, item_categorical: torch.Tensor, item_dense: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.item_embedding(item_categorical[:, 0]), dim=1)

    def forward(self, user_categorical: torch.Tensor, user_dense: torch.Tensor, item_categorical: torch.Tensor, item_dense: torch.Tensor) -> torch.Tensor:
        user_vector = self.user_vector(user_categorical, user_dense)
        item_vector = self.item_vector(item_categorical, item_dense)
        return torch.sum(user_vector * item_vector, dim=1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        user_cardinalities: list[int],
        item_cardinalities: list[int],
        user_dense_dim: int,
        item_dense_dim: int,
        embedding_dim: int,
        tower_dims: list[int],
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.user_tower = TowerBackbone(user_cardinalities, user_dense_dim, embedding_dim, tower_dims, output_dim, dropout)
        self.item_tower = TowerBackbone(item_cardinalities, item_dense_dim, embedding_dim, tower_dims, output_dim, dropout)

    def user_vector(self, user_categorical: torch.Tensor, user_dense: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_categorical, user_dense)

    def item_vector(self, item_categorical: torch.Tensor, item_dense: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_categorical, item_dense)

    def forward(self, user_categorical: torch.Tensor, user_dense: torch.Tensor, item_categorical: torch.Tensor, item_dense: torch.Tensor) -> torch.Tensor:
        user_vector = self.user_vector(user_categorical, user_dense)
        item_vector = self.item_vector(item_categorical, item_dense)
        return torch.sum(user_vector * item_vector, dim=1)


def build_model(name: str, config: dict[str, Any], user_cardinalities: list[int], item_cardinalities: list[int], user_dense_dim: int, item_dense_dim: int) -> nn.Module:
    output_dim = int(config.get("output_dim", 32))
    if name == "matrix_factorization":
        return MatrixFactorizationModel(user_cardinalities[0], item_cardinalities[0], output_dim)
    if name == "two_tower":
        return TwoTowerModel(
            user_cardinalities=user_cardinalities,
            item_cardinalities=item_cardinalities,
            user_dense_dim=user_dense_dim,
            item_dense_dim=item_dense_dim,
            embedding_dim=int(config.get("embedding_dim", 16)),
            tower_dims=list(config.get("tower_dims", [64, 32])),
            output_dim=output_dim,
            dropout=float(config.get("dropout", 0.1)),
        )
    raise ValueError(f"Unsupported retrieval model: {name}")
