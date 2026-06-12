from dataclasses import dataclass
import numpy as np
import scipy
import torch

from libam.algorithms.algorithm import AlignAlgorithm
from libam.algorithms.utils.matrix_utils import normout_rowstochastic
from libam.graph.graph_pair import GraphPair


#original code from https://github.com/nassarhuda/NetworkAlignment.jl/blob/master/src/NSD.jl

@dataclass
class NSD(AlignAlgorithm):
    pair: GraphPair
    alpha: float
    iters: int

    @property
    def name(self):
        return "NSD"

    def _align(self) -> np.ndarray | torch.Tensor | scipy.sparse.csr_matrix:
        src = self.pair.src_adjacency
        tar = self.pair.tar_adjacency

        x = nsd(
            src, tar, self.alpha, self.iters,
            np.ones((src.shape[0], 1)),
            np.ones((tar.shape[0], 1)),
        )

        return x

def nsd(A, B, alpha, iters, Zvecs, Wvecs):
    dtype = np.float32
    A = normout_rowstochastic(A).T.astype(dtype)
    B = normout_rowstochastic(B).T.astype(dtype)
    Zvecs = Zvecs.astype(dtype=dtype)
    Wvecs = Wvecs.astype(dtype=dtype)
    nB = np.shape(B)[0]
    nA = np.shape(A)[0]

    Sim = np.zeros((nA, nB), dtype=dtype)
    for i in range(np.shape(Zvecs)[1]):
        z = Zvecs[:, i]
        w = Wvecs[:, i]
        z = z / sum(z)
        w = w / sum(w)
        Z = np.zeros((iters + 1, nA), dtype=dtype)  # A
        W = np.zeros((iters + 1, nB), dtype=dtype)  # B
        W[0] = w
        Z[0] = z
        for k in range(1, iters + 1):
            np.dot(A, Z[k-1], out=Z[k])
            np.dot(B, W[k-1], out=W[k])

        factor = 1.0 - alpha
        for k in range(iters + 1):
            if k == iters:
                W[iters] *= alpha ** iters
            else:
                W[k] *= factor
                factor *= alpha
            intervals = 4
            for i in range(intervals):
                start = i * nA // intervals
                end = (i+1) * nA // intervals
                Sim[:, start:end] += np.dot(
                    Z[k].reshape(-1, 1), W[k][start:end].reshape(1, -1)
                )

    return Sim

