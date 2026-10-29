"""Geo-cell classifier: a linear layer trained on frozen CLIP embeddings.

Singapore is cut into square cells; a softmax regression learns which cell an embedding
belongs to, and predicts the mean location of that cell's training photos. Unlike kNN it
learns which embedding directions actually matter for location (vegetation, building
style, road furniture) rather than overall visual similarity. Trains in about a minute on
CPU from an existing index; CLIP itself is not touched.
"""

from __future__ import annotations

import numpy as np


class GeoCellClassifier:
    def __init__(self, cell_deg: float = 0.01, l2: float = 1e-3, steps: int = 300, lr: float = 0.01):
        self.cell_deg = cell_deg
        self.l2 = l2
        self.steps = steps
        self.lr = lr

    def fit(self, X: np.ndarray, coords: np.ndarray, log=None) -> "GeoCellClassifier":
        keys = np.floor(coords / self.cell_deg).astype(np.int64)
        _, y = np.unique(keys, axis=0, return_inverse=True)
        y = y.ravel()
        n_cells = int(y.max()) + 1
        self.centroids = np.zeros((n_cells, 2))
        np.add.at(self.centroids, y, coords)
        self.centroids /= np.bincount(y, minlength=n_cells)[:, None]

        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        Xs = ((X - self.mu) / self.sd).astype(np.float32)
        n, d = Xs.shape
        onehot = np.zeros((n, n_cells), dtype=np.float32)
        onehot[np.arange(n), y] = 1

        W = np.zeros((d, n_cells), dtype=np.float32)
        b = np.zeros(n_cells, dtype=np.float32)
        mW, vW, mb, vb = (np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b))
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.steps + 1):
            p = _softmax(Xs @ W + b)
            g = (p - onehot) / n
            gW = Xs.T @ g + self.l2 * W
            gb = g.sum(axis=0)
            for param, grad, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
                m *= beta1
                m += (1 - beta1) * grad
                v *= beta2
                v += (1 - beta2) * grad**2
                param -= self.lr * (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps)
            if log and t % 100 == 0:
                loss = -np.log(p[np.arange(n), y] + 1e-9).mean()
                log(f"  geocell {self.cell_deg}: step {t}, loss {loss:.3f}, {n_cells} cells")
        self.W, self.b = W, b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = ((np.atleast_2d(X) - self.mu) / self.sd).astype(np.float32)
        return _softmax(Xs @ self.W + self.b)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(coords, confidence): centroid of the most likely cell for each row of X."""
        p = self.predict_proba(X)
        best = p.argmax(axis=1)
        return self.centroids[best], p[np.arange(len(best)), best]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
