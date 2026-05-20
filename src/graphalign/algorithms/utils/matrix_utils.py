"""Shared matrix utility functions used across multiple graph alignment algorithms."""

import numpy as np
import numpy.linalg as lg
import scipy.sparse as sps
import torch


def doubly_stochastic(P: torch.Tensor, tau: float, it: int) -> torch.Tensor:
    """Project a matrix onto the set of doubly stochastic matrices.

    Uses the log-sum-exp for numerical stability.

    Args:
        P: Input tensor to project.
        tau: Temperature parameter controlling sharpness.
        it: Number of Sinkhorn iterations.

    Returns:
        A doubly stochastic tensor of the same shape as P.
    """
    A = P / tau
    for _ in range(it):
        A = A - A.logsumexp(dim=1, keepdim=True)
        A = A - A.logsumexp(dim=0, keepdim=True)
    return torch.exp(A)


def regularise_invert_one(x: np.ndarray, alpha: float, ones: bool) -> np.ndarray:
    """Compute a regularised (pseudo-)inverse of a matrix.

    Args:
        x: Input square matrix.
        alpha: Regularisation strength added to the diagonal.
        ones: If True, also adds a rank-1 all-ones correction term before inverting.
              If False, computes the Moore-Penrose pseudoinverse and adds alpha * I.

    Returns:
        The regularised inverse as a numpy array.
    """
    if ones:
        return lg.inv(x + alpha * np.eye(len(x)) + np.ones([len(x), len(x)]) / len(x))
    return lg.pinv(x) + alpha * np.eye(len(x))


def regularise_and_invert(x: np.ndarray, y: np.ndarray, alpha: float, ones: bool) -> list[np.ndarray]:
    """Apply regularised inversion to a pair of matrices.

    Args:
        x: First input matrix.
        y: Second input matrix.
        alpha: Regularisation strength.
        ones: Whether to include the rank-1 all-ones correction (see regularise_invert_one).

    Returns:
        A list [x_reg, y_reg] of regularised inverse matrices.
    """
    return [regularise_invert_one(x, alpha, ones), regularise_invert_one(y, alpha, ones)]


def normout_rowstochastic(P: np.ndarray | sps.spmatrix) -> np.ndarray:
    """
    Row-normalise a (sparse or dense) matrix so each row sums to 1

    :param P: Matrix, either dense numpy array or a scipy sparse matrix
    :return: Dense numpy arrray with rows summing to 1 (all zero will remain as is)
    """


    n = P.shape[0]
    m = P.shape[1]
    pi, pj, pv = sps.find(P)
    row_sums = np.asarray(P.sum(axis=1)).ravel()

    safe_sums = np.where(row_sums == 0, 1.0, row_sums)
    pv = pv / safe_sums[pi]
    return sps.csc_matrix((pv, (pi, pj)), shape=(n, m)).toarray()


def to_torch(x: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy array to a float64 torch tensor

    :param x: Input numpy array
    :return: torch.DoubleTensor with the same data
    """
    return torch.from_numpy(x.astype(np.float64))


def node_to_degree(G_degree, SET):
    SET = list(SET)
    SET = sorted([G_degree[x] for x in SET])
    return SET


