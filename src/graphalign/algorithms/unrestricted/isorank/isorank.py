from dataclasses import dataclass

import numpy as np
from numpy import inf, nan
import scipy.sparse as sps
import scipy
# from algorithms import bipartiteMatching
# from algorithms.NSD.NSD import fast2, findnz1
# from data import similarities_preprocess, ReadFile
# from evaluation import evaluation
# from evaluation.evaluation import check_with_identity
# from evaluation.evaluation_design import remove_edges_directed
# from experiment.similarities_preprocess import create_L
from math import floor, log2

from scipy.sparse import csr_matrix

from graphalign import GraphPair
from graphalign.algorithms.algorithm import Algorithm


def create_L(A, B, lalpha=1, mind=None, weighted=True):
    n = A.shape[0]
    m = B.shape[0]

    if lalpha is None:
        return sps.csr_matrix(np.ones((n, m)))

    a = A.sum(1)
    b = B.sum(1)

    # a_p = [(i, m[0,0]) for i, m in enumerate(a)]
    a_p = list(enumerate(a))
    a_p.sort(key=lambda x: x[1])

    # b_p = [(i, m[0,0]) for i, m in enumerate(b)]
    b_p = list(enumerate(b))
    b_p.sort(key=lambda x: x[1])

    ab_m = [0] * n
    s = 0
    e = floor(lalpha * log2(m))
    for ap in a_p:
        while(e < m and
              abs(b_p[e][1] - ap[1]) <= abs(b_p[s][1] - ap[1])
              ):
            e += 1
            s += 1
        ab_m[ap[0]] = [bp[0] for bp in b_p[s:e]]

    # print(ab_m)

    li = []
    lj = []
    lw = []
    for i, bj in enumerate(ab_m):
        for j in bj:
            # d = 1 - abs(a[i]-b[j]) / a[i]
            d = 1 - abs(a[i]-b[j]) / max(a[i], b[j])
            if mind is None:
                if d > 0:
                    li.append(i)
                    lj.append(j)
                    lw.append(d)
            else:
                li.append(i)
                lj.append(j)
                lw.append(mind if d <= 0 else d)
                # lw.append(0.0 if d <= 0 else d)
                # lw.append(d)

                # print(len(li))
                # print(len(lj))
                # print(len(lj))

    return sps.csr_matrix((lw, (li, lj)), shape=(n, m))


@dataclass
class Isorank(Algorithm):
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

    def evaluate(self) -> np.ndarray:
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