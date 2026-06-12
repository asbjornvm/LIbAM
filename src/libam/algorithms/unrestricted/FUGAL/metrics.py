#Fugal Algorithm was provided by anonymous authors.
import math

import numpy as np


def EC(A: np.ndarray, B: np.ndarray, ma: np.ndarray, mb: np.ndarray) -> float:
    adj1 = A[ma][:, ma]
    adj2 = B[mb][:, mb]
    comb = adj1 + adj2

    intersection = np.sum(comb == 2)

    return intersection / np.sum(A == 1)


def ICS(A: np.ndarray, B: np.ndarray, ma: np.ndarray, mb: np.ndarray) -> float:
    adj1 = A[ma][:, ma]
    adj2 = B[mb][:, mb]
    comb = adj1 + adj2

    intersection = np.sum(comb == 2)
    induced = np.sum(adj2 == 1)

    return intersection / induced


def S3(A: np.ndarray, B: np.ndarray, ma: np.ndarray, mb: np.ndarray) -> float:
    adj1 = A[ma][:, ma]
    adj2 = B[mb][:, mb]
    comb = adj1 + adj2

    intersection = np.sum(comb == 2)
    induced = np.sum(adj2 == 1)
    denom = np.sum(A == 1) + induced - intersection

    return intersection / denom


def jacc(A: np.ndarray, B: np.ndarray, ma: np.ndarray, mb: np.ndarray) -> float:
    adj1 = A[ma][:, ma]
    adj2 = B[mb][:, mb]
    comb = adj1 + adj2

    intersection = np.sum(comb == 2)
    union = np.sum(A == 1) + np.sum(B == 1) - intersection

    return intersection / union

def eval_align(ma: np.ndarray, mb: np.ndarray, gmb: np.ndarray) -> tuple[float, float, np.ndarray]:

    try:
        gmab = np.arange(gmb.size)
        gmab[ma] = mb
        gacc = np.mean(gmb == gmab)

        mab = gmb[ma]
        acc = np.mean(mb == mab)

    except Exception:
        mab = np.zeros(mb.size, int) - 1
        gacc = acc = -1.0
    alignment = np.array([ma, mb, mab]).T
    alignment = alignment[alignment[:, 0].argsort()]
    return gacc, acc, alignment


def ged(A: np.ndarray, B: np.ndarray, ma: np.ndarray, mb: np.ndarray) -> float:
    n = ma.size
    P = np.zeros((n, n))
    for i in range(n):
        P[i][mb[i]] = 1
    X = np.matmul(A, P) - np.matmul(P, B)
    norm = np.linalg.norm(X, 'fro')
    return norm*norm/2

def ged_rmse(v1: list[float], v2: list[float]) -> float:
    n = len(v1)
    res = 0
    for i in range(n):
        res += (v1[i] - v2[i])**2
    res /= n
    return math.sqrt(res)

def avg(lb: list[float], ub: list[float]) -> list[float]:
    res = []
    for i in range(len(lb)):
        res.append((lb[i] + ub[i])/2)
    return res

def rmse(lb: list[float], ub: list[float], pred: list[float]) -> float:
    truth = avg(lb, ub)
    rmse_error = ged_rmse(pred, truth)
    return rmse_error
