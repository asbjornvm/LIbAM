from dataclasses import dataclass
import numpy as np
import time

import ot
import scipy
import networkx as nx
import torch

from libam.graph.graph_pair import GraphPair
from libam.algorithms.algorithm import AlignAlgorithm


def get_prod_rwr(L1, L2, nodeCost, H, beta, gamma, prodRwrIter):

    eps = 1e-2
    nx, ny = H.T.shape
    HInd = np.nonzero(H.T)
    crossCost = np.zeros((nx, ny))
    i = 0
    res = np.inf
    
    while i < prodRwrIter and res > eps:
        rwCost_old = crossCost.copy()
        crossCost = (1 + gamma * beta) * nodeCost + (1 - beta) * gamma * L1.dot(crossCost).dot(L2.T)
        crossCost[HInd] = 0
        res = np.max(np.abs(crossCost - rwCost_old))
        i += 1
    
    crossCost = (1 - gamma) * crossCost
    crossCost[HInd] = 0
    
    return crossCost


def get_intra_cost(V):

    # Cosine dissimilarity
    row = np.where(np.sum(np.abs(V), axis=1) == 0)[0]
    _, d = V.shape
    V = V / np.sqrt(np.sum(V * V, axis=1, keepdims=True))
    V[row, :] = np.sqrt(1 / d)
    intraCost = np.exp(-np.dot(V, V.T))
    
    return intraCost


def get_cross_cost(V1, V2, H):

    row1 = np.where(np.sum(np.abs(V1), axis=1) == 0)[0]
    row2 = np.where(np.sum(np.abs(V2), axis=1) == 0)[0]
    _, d = V1.shape
    V1 = V1 / np.sqrt(np.sum(V1 * V1, axis=1, keepdims=True))
    V2 = V2 / np.sqrt(np.sum(V2 * V2, axis=1, keepdims=True))
    V1[row1, :] = np.sqrt(1 / d)
    V2[row2, :] = np.sqrt(1 / d)
    crossCost = np.exp(-np.dot(V1, V2.T))
    HInd = np.nonzero(H.T)
    crossCost[HInd] = 0
    
    return crossCost

def get_sep_rwr(T1, T2, H, beta, sepRwrIter):

    eps = 1e-5
    anchor1, anchor2 = np.nonzero(H.T)
    n1 = len(T1)
    n2 = len(T2)
    a = len(anchor1)

    # construct anchor links
    e1 = np.zeros((n1, a))
    e2 = np.zeros((n2, a))
    e1[anchor1, np.arange(a)] = 1
    e2[anchor2, np.arange(a)] = 1

    r1 = np.zeros((n1, a))
    r2 = np.zeros((n2, a))
    
    i = 0
    res = np.inf
    
    while i < sepRwrIter and res > eps:
        r1_old = r1.copy()
        r2_old = r2.copy()
        r1 = (1 - beta) * T1.dot(r1) + beta * e1
        r2 = (1 - beta) * T2.dot(r2) + beta * e2
        res = max(np.max(np.abs(r1 - r1_old)), np.max(np.abs(r2 - r2_old)))
        i += 1

    return r1, r2


def cal_trans(A, V):
    
    n = len(A)
    T = np.copy(A)
    
    if V.size == 0:  # without node attribute
        V = np.ones((n, 1))
    
    V = V / np.sqrt(np.sum(V * V, axis=1, keepdims=True))
    sim = np.dot(V, V.T)
    T = sim * A
    
    for row in range(n):
        k = np.where(A[row, :])[0]  # Find the index which value is not 0.
        T[row, k] = softmax(T[row, k])
    
    return T

def softmax(x):
    """Compute softmax values for each row of matrix x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_cost(A1, A2, H, rwrIter, rwIter, alpha, beta, gamma):

    tCostStart = time.time()
    
    # calculate rwr
    T1 = cal_trans(A1, np.array([]))
    T2 = cal_trans(A2, np.array([]))
    
    # cross/intra-graph cost based on node attributes
    V1 = np.ones((len(A1), 1))#rwr1
    V2 = np.ones((len(A2), 1))#rwr2
    
    crossC = get_cross_cost(V1, V2, H)
    
    intraC1 = A1#np.multiply(A1, get_intra_cost(V1))
    intraC2 = A2#np.multiply(A2, get_intra_cost(V2))
    crossC = crossC# + alpha * rwrCost
    L1 = A1 / np.sum(A1, axis=1, keepdims=True)  # Laplacian matrix
    L2 = A2 / np.sum(A2, axis=1, keepdims=True)
    crossC = get_prod_rwr(L1, L2, crossC, H, beta, gamma, rwIter) 
    tCostEnd = time.time()
    print("time for cost matrix: %.2fs" % (tCostEnd - tCostStart))

    return crossC, intraC1, intraC2

def cpot(L1, L2, crossC, intraC1, intraC2, inIter, outIter, H, l1, l2, l3, l4):

    nx, ny = crossC.shape
    l4 = l4 * nx * ny
    eps = 0
    
    # Define initial matrix values
    a = np.ones(nx) / nx
    b = np.ones(ny) / ny
    r = np.ones((nx, 1)) / nx
    c = np.ones((1, ny)) / ny
    l = l1 + l2 + l3
    
    T = np.ones((nx, ny)) / (nx * ny)  # initial alignment score
    H = H.T + np.ones((nx, ny)) / ny  # prior knowledge, in case of zero
    
    # Functions for OT
    # mina = lambda H, epsilon: -epsilon * np.log(np.sum(a * np.exp(-H / epsilon), axis=1))
    # minb = lambda H, epsilon: -epsilon * np.log(np.sum(b * np.exp(-H / epsilon), axis=0))
    # minaa = lambda H, epsilon: mina(H - np.min(H, axis=1, keepdims=True), epsilon) + np.min(H, axis=1)
    # minbb = lambda H, epsilon: minb(H - np.min(H, axis=0, keepdims=True), epsilon) + np.min(H, axis=0)
    
    def mina(H, epsilon, a):
        return -epsilon * np.log(np.sum(a * np.exp(-H / epsilon), axis=0))

    def minb(H, epsilon, b):
        print(H.shape, b.shape)
        return -epsilon * np.log(np.sum(b * np.exp(-H / epsilon), axis=1))

    # Define minaa and minbb functions
    def minaa(H, epsilon, a):
        min_H = np.min(H, axis=0)
        return mina(H - min_H, epsilon, a) + min_H

    def minbb(H, epsilon, b):
        min_H = np.min(H, axis=1)
        return minb(H - min_H[:, np.newaxis], epsilon, b) + min_H

    temp1 = 0.5 * intraC1 ** 2 @ r @ np.ones((1, ny)) + 0.5 * np.ones((nx, 1)) @ c @ (intraC2 ** 2).T  # const term in GWD
    
    i = 0
    res = np.inf
    resRecord = []
    WRecord = []
    tIpotStart = time.time()
    
    outIter = min(outIter, np.max(crossC) * np.log(max(nx, ny)) * np.power(np.spacing(1), -3))
    # outIter = min(outIter, np.max(crossC) * np.log(max(nx, ny)) * eps ** (-3))
    
    while i < outIter:
        #print("Sum: ", np.sum(T))
        T_old = T
        CGW = temp1 - intraC1 @ T @ intraC2.T  # Frobenius GWD
        C = crossC - l2 * np.log(L1 @ T @ L2.T) - l3 * np.log(H) + l4 * CGW  # modified cost
        
        if i == 0:
            C_old = C
        else:
            W_old = np.sum(T * C_old)
            W = np.sum(T * C)
            if W <= W_old:
                C_old = C
            else:
                C = C_old
        
        Q = C - l1 * np.log(T)

        X = ot.sinkhorn(a, b, Q, reg=l, numItermax=inIter)

        T = 0.05 * T_old + 0.95 * X#np.matmul(np.matmul(np.diag(np.squeeze(r)), np.exp((a + b - Q) / l)), np.diag(np.squeeze(c)))  # soft update
        res = np.sum(np.abs(T - T_old))
        resRecord.append(res)
        WRecord.append(np.sum(T * C))
        i += 1
    
    tIpotEnd = time.time() - tIpotStart
    print(f"time for optimization: {tIpotEnd:.2f}s")
    
    return T, WRecord, resRecord


@dataclass
class Parrot(AlignAlgorithm):
    pair: GraphPair
    sepRwrIter: int
    prodRwrIter: int
    alpha: float
    beta: float
    gamma: float
    in_iter: int
    out_iter: int
    l1: float
    l2: float
    l3: float
    l4: float

    @property
    def name(self):
        return "Parrot"

    def _align(self) -> np.ndarray | torch.Tensor | scipy.sparse.csr_matrix:
        a1 = self.pair.src_adjacency
        a2 = self.pair.tar_adjacency

        sep_rwr_iter = self.sepRwrIter
        prod_rwr_iter = self.prodRwrIter
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        in_iter = self.in_iter
        out_iter = self.out_iter
        l1 = self.l1
        l2 = self.l2
        l3 = self.l3
        l4 = self.l4

        n1 = a1.shape[0]
        H=np.zeros((n1,n1))
        nx, ny = H.T.shape
        row1 = np.where(np.sum(a1, axis=1) == 0)[0]
        row2 = np.where(np.sum(a2, axis=1) == 0)[0]
        a1[row1, :] = np.ones((len(row1), nx))
        a2[row2, :] = np.ones((len(row2), ny))
        L1 = a1 / np.sum(a1, axis=1, keepdims=True) # Laplacian matrix
        L2 = a2 / np.sum(a2, axis=1, keepdims=True)

        # calculate cross-graph/intra-graph cost if not pre-computed
        crossC, intraC1, intraC2 = get_cost(a1, a2, H, sep_rwr_iter, prod_rwr_iter, alpha, beta, gamma)
        #print(np.max(crossC), np.min(crossC), np.sum(crossC))
        T, W, res = cpot(L1, L2, crossC, intraC1, intraC2, in_iter, out_iter, H, l1, l2, l3, l4)

        return T

