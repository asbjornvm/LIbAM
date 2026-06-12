"""Shared matrix utility functions used across multiple graph alignment algorithms."""
import typing

import numpy as np
import numpy.linalg as lg
import scipy
import torch


def doubly_stochastic_same_type(p: np.ndarray | torch.Tensor, iterations: int) -> np.ndarray:
    """
    Project a matrix onto the set of doubly stochastic matrices.

    Can possibly create nan if there are negative values in p, should not happen in practice.

    :param p: input matrix or tensor
    :param iterations: Number of Sinkhorn iterations
    :return: Doubly stochastic numpy matrix of the same shape as p.
    """
    if isinstance(p, torch.Tensor):
        a = p.double()
        for _ in range(iterations):
            a = a - torch.logsumexp(a, dim=1, keepdim=True)
            a = a - torch.logsumexp(a, dim=0, keepdim=True)
        return torch.exp(a)
    else:
        a = p.astype(np.float64)
        for _ in range(iterations):
            row_max = np.max(a, axis=1, keepdims=True)
            a -= row_max + np.log(np.sum(np.exp(a - row_max), axis=1, keepdims=True))
            col_max = np.max(a, axis=0, keepdims=True)
            a -= col_max + np.log(np.sum(np.exp(a - col_max), axis=0, keepdims=True))
        return np.exp(a)


def doubly_stochastic(p: np.ndarray | torch.Tensor | scipy.sparse.csr_matrix, iterations: int) -> np.ndarray:
    """
    Project a matrix onto the set of doubly stochastic matrices.

    Can possibly create nan if there are negative values in p, should not happen in practice.

    :param p: input matrix or tensor
    :param iterations: Number of Sinkhorn iterations
    :return: Doubly stochastic numpy matrix of the same shape as p.
    """
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    elif isinstance(p, scipy.sparse.csr_matrix):
        p = p.toarray()

    a = p.astype(np.float64)
    for _ in range(iterations):
        row_max = np.max(a, axis=1, keepdims=True)
        a -= row_max + np.log(np.sum(np.exp(a - row_max), axis=1, keepdims=True))
        col_max = np.max(a, axis=0, keepdims=True)
        a -= col_max + np.log(np.sum(np.exp(a - col_max), axis=0, keepdims=True))
    return np.exp(a)


def regularise_invert_one(x: np.ndarray, alpha: float, ones: bool) -> np.ndarray:
    """
    Compute a regularised (pseudo-)inverse of a matrix

    :param x:
    :param alpha:
    :param ones:
    :return:
    """
    if ones:
        return lg.inv(x + alpha * np.eye(len(x)) + np.ones([len(x), len(x)]) / len(x))
    return lg.pinv(x) + alpha * np.eye(len(x))


def regularise_and_invert(x: np.ndarray, y: np.ndarray, alpha: float, ones: bool) -> list[np.ndarray]:
    """
    Apply regularised inversion to a pair of matrices

    :param x: First input matrix
    :param y: Second input matrix
    :param alpha:
    :param ones:
    :return: A list [x_reg, y_reg] of regularised inverse matrices.
    """
    x_reg = regularise_invert_one(x, alpha, ones)
    y_reg = regularise_invert_one(y, alpha, ones)
    return [x_reg, y_reg]


def normout_rowstochastic(P: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
    """
    Row-normalise a (sparse or dense) matrix so each row sums to 1

    :param P: Matrix, either dense numpy array or a scipy sparse matrix
    :return: Dense numpy arrray with rows summing to 1 (all zero will remain as is)
    """


    n = P.shape[0]
    m = P.shape[1]
    pi, pj, pv = scipy.sparse.find(P)
    row_sums = np.asarray(P.sum(axis=1)).ravel()

    safe_sums = np.where(row_sums == 0, 1.0, row_sums)
    pv = pv / safe_sums[pi]
    return scipy.sparse.csc_matrix((pv, (pi, pj)), shape=(n, m)).toarray()


def to_torch(x: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy array to a float64 torch tensor

    :param x: Input numpy array
    :return: torch.DoubleTensor with the same data
    """
    return torch.from_numpy(x.astype(np.float64))


def node_to_degree(G_degree, SET):
    """
    Map a set of node IDs to their corresponding degrees

    :param G_degree:
    :param SET:
    :return:
    """

    SET = list(SET)
    SET = sorted([G_degree[x] for x in SET])
    return SET


