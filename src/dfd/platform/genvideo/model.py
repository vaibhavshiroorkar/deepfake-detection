"""Small MLP head stored as plain arrays (no pickle): standardise, hidden layers with ReLU, sigmoid output."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Head:
    mean: np.ndarray
    std: np.ndarray
    W: list
    b: list
    threshold: float
    meta: dict

    def score(self, x) -> np.ndarray:
        h = (np.atleast_2d(x).astype(np.float64) - self.mean) / np.maximum(self.std, 1e-8)
        for i in range(len(self.W) - 1):
            h = np.maximum(h @ self.W[i] + self.b[i], 0.0)
        z = (h @ self.W[-1] + self.b[-1]).ravel()
        return 1.0 / (1.0 + np.exp(-z))


def save_head(path, mean, std, coefs, intercepts, threshold: float, meta: dict) -> None:
    arrs = {"mean": np.asarray(mean), "std": np.asarray(std), "threshold": np.array([threshold]), "n_layers": np.array([len(coefs)]),
            "meta_json": np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)}
    for i, (w, b) in enumerate(zip(coefs, intercepts)):
        arrs[f"W{i}"], arrs[f"b{i}"] = np.asarray(w), np.asarray(b)
    np.savez(path, **arrs)


def load_head(path) -> Head:
    z = np.load(path, allow_pickle=False)
    n = int(z["n_layers"][0])
    return Head(z["mean"], z["std"], [z[f"W{i}"] for i in range(n)], [z[f"b{i}"] for i in range(n)], float(z["threshold"][0]),
                json.loads(bytes(z["meta_json"]).decode()))
