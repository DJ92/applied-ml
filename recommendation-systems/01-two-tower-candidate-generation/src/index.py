from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class NumpyANNIndex:
    def __init__(self, item_ids: list[str], item_vectors: np.ndarray) -> None:
        self.item_ids = np.asarray(item_ids)
        self.item_vectors = np.asarray(item_vectors, dtype="float32")

    def search(self, query_vectors: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
        scores = query_vectors @ self.item_vectors.T
        indices = np.argsort(scores, axis=1)[:, ::-1][:, :topk]
        top_scores = np.take_along_axis(scores, indices, axis=1)
        top_items = self.item_ids[indices]
        return top_items, top_scores

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump({"item_ids": self.item_ids.tolist(), "item_vectors": self.item_vectors}, handle)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NumpyANNIndex":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        return cls(payload["item_ids"], np.asarray(payload["item_vectors"], dtype="float32"))


class FaissANNIndex(NumpyANNIndex):
    def __init__(self, item_ids: list[str], item_vectors: np.ndarray) -> None:
        import faiss

        super().__init__(item_ids, item_vectors)
        self._faiss = faiss
        self.index = faiss.IndexFlatIP(item_vectors.shape[1])
        self.index.add(self.item_vectors)

    def search(self, query_vectors: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
        scores, indices = self.index.search(query_vectors.astype("float32"), topk)
        return self.item_ids[indices], scores


def build_best_available_index(item_ids: list[str], item_vectors: np.ndarray) -> NumpyANNIndex:
    try:
        return FaissANNIndex(item_ids, item_vectors)
    except Exception:
        return NumpyANNIndex(item_ids, item_vectors)
