from dataclasses import dataclass

import numpy as np
import torch
from numpy import inf, nan
import scipy.sparse as sps
import scipy
from math import floor, log2

from libam.graph._similarities_preprocess import create_L
from libam.graph.graph_pair import GraphPair
from libam.algorithms.algorithm import AlignAlgorithm


@dataclass
class Isorank(AlignAlgorithm):
    pair: GraphPair
    L: object
    lalpha: float
    weighted: bool
    alpha: float
    tol: float
    max_iter: int

    @property
    def name(self) -> str:
        return "Isorank"

    def _align(self) -> np.ndarray | torch.Tensor | scipy.sparse.csr_matrix:
        dtype = np.float32
        Src = self.pair.src_adjacency
        Tar = self.pair.tar_adjacency
        L = self.L

        if self.lalpha is not None:
            L = create_L(Src, Tar, lalpha=self.lalpha, weighted=self.weighted).toarray().astype(dtype)

        n1 = Tar.shape[0]
        n2 = Src.shape[0]

        # normalize the adjacency matrices
        d1 = 1 / Tar.sum(axis=1)
        d2 = 1 / Src.sum(axis=1)

        d1[d1 == inf] = 0
        d2[d2 == inf] = 0
        d1 = d1.reshape(-1, 1)
        d2 = d2.reshape(-1, 1)

        W1 = d1*Tar
        W2 = d2*Src
        W2aT = (self.alpha * W2.T).astype(dtype)
        K = ((1-self.alpha) * L).astype(dtype)
        W1 = W1.astype(dtype)

        S = np.ones((n2, n1), dtype=dtype) / (n1 * n2)  # Map target to source
        # IsoRank Algorithm in matrix form
        for it in range(1, self.max_iter + 1):
            prev = S.flatten()
            if self.alpha is None:
                S = W2.T.dot(S).dot(W1)
            else:
                S = W2aT.dot(S).dot(W1) + K
            delta = np.linalg.norm(S.flatten()-prev, 2)
            if delta < self.tol:
                break

        return S