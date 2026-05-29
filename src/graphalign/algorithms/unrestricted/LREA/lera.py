from dataclasses import dataclass

import numpy as np
import scipy
import torch
from numpy.linalg import svd
#original code https://github.com/nassarhuda/lowrank_spectral
from . import decomposeX, newbound_methods
from graphalign.algorithms.algorithm import Algorithm
from graphalign.graph.graph_pair import GraphPair


# eigenalign
@dataclass
class Lera(Algorithm):
    pair: GraphPair
    iters: int
    method: str
    b_match: int
    default_params: bool

    @property
    def name(self):
        return "LERA"

    def _evaluate(self) -> np.ndarray | torch.Tensor | scipy.sparse.csr_matrix:
        iters = self.iters
        method = self.method
        b_match = self.b_match
        default_params = self.default_params

        src = self.pair.src_adjacency
        tar = self.pair.tar_adjacency

        s1, s2, s3 = find_parameters(src, tar)

        if not default_params:
            s1 += 100
            s2 += 10
            s3 += 5
        c1 = s1 + s2 - 2 * s3
        c2 = s3 - s2
        c3 = s2
        Uk, Vk, Wk, W1, W2 = decomposeX.decomposeX_balance_allfactors(
            src, tar, iters + 1, c1, c2, c3)  # okay
        Un, Vn = split_balanced_decomposition(Uk, Wk, Vk)  # okay
        timematching = 0
        nA = len(src[0])
        nB = len(tar[0])

        if method == "lowrank_svd_union":

            U, S, Vtemp = np.linalg.svd(Wk)
            V = Vtemp.transpose()
            U1 = np.dot(np.dot(Uk, U), np.diag(np.sqrt(S)))
            V1 = np.dot(np.dot(Vk, V), np.diag(np.sqrt(S)))
            # X = newbound_methods.newbound_rounding_lowrank_evaluation_relaxed(U1, V1, bmatch) * (10 ** 8)  # 1
            X, nzi, nzj, nzv = newbound_methods.newbound_rounding_lowrank_evaluation_relaxed(U1, V1, b_match)  # alternative
        else:
            print(
                "method should be one of the following: (1)eigenalign,(2)lowrank_unbalanced_best, (3)lowrank_unbalanced_union,(4)lowrank_balanced_best, (5)lowrank_balanced_union,(6)lowrank_Wkdecomposed_best, (7)lowrank_Wkdecomposed_union")
        return scipy.sparse.csr_matrix((nzv, (nzi, nzj)))


def find_parameters(A, B):
    nB = len(B[0])
    nA = len(A[0])
    myalpha = (nB ** 2 - np.sum(B)) / np.sum(B) + \
        (nA ** 2 - np.sum(A)) / np.sum(A)+1
    myeps = 0.001
    s1 = myalpha + myeps
    s2 = 1 + myeps
    s3 = myeps
    return s1, s2, s3


def split_balanced_decomposition(Uk, Wk, Vk):
    P, L, U = scipy.linalg.lu(Wk, False)
    Ud = np.diag(np.sqrt(abs(np.diag(U))))
    L2 = np.dot(L, Ud)
    Utemp = np.sqrt(np.diag(U))
    Utemp2 = np.divide(1, Utemp)
    U2 = np.dot(np.diag(Utemp2), U)
    Un = np.dot(Uk, L2)
    Vn = np.dot(Vk, U2.transpose())

    return Un, Vn


def split_svd(Uk, Wk, Vk):
    U, S, V = svd(Wk)
    D = np.diag(np.sqrt(S))
    Unew = Uk @ U @ D
    Vnew = Vk @ V @ D
    return Unew, Vnew
