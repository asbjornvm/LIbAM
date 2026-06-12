import numpy as np
import scipy
import scipy.sparse as sps
from numpy import linalg as LA

import libam
from libam.graph.graph_pair import GraphPair


def EC(A, B, ma, mb):
    """Edge Correctness of an alignment.
    
    :param A: Source adjacency matrix.
    :param B: Target adjacency matrix.
    :param ma: Matched source node indices.
    :param mb: Matched target node indices.
    """
    adj1 = A[ma][:, ma]
    adj2 = B[mb][:, mb]
    comb = adj1 + adj2

    intersection = np.sum(comb == 2)

    return intersection / np.sum(A == 1)

def ICS(A, B, ma, mb):
    """Induced Conserved Structure score of an alignment.

    :param A: Source adjacency matrix.
    :param B: Target adjacency matrix.
    :param ma: Matched source node indices.
    :param mb: Matched target node indices.
    """
    adj1 = A[ma][:, ma]
    adj2 = B[mb][:, mb]
    comb = adj1 + adj2

    intersection = np.sum(comb == 2)
    induced = np.sum(adj2 == 1)

    return intersection / induced

def S3(A, B, ma, mb):
    """
    param A: Source adjacency matrix.
    :param B: Target adjacency matrix.
    :param ma: Matched source node indices.
    :param mb: Matched target node indices.
    """
    adj1 = A[ma][:, ma]
    adj2 = B[mb][:, mb]
    comb = adj1 + adj2

    intersection = np.sum(comb == 2)
    induced = np.sum(adj2 == 1)
    denom = np.sum(A == 1) + induced - intersection

    return intersection / denom


def score_MNC(adj1, adj2, countera, counterb):
    try:
        mnc = 0
        if sps.issparse(adj1):
            adj1 = adj1.toarray()
        if sps.issparse(adj2):
            adj2 = adj2.toarray()
        for cri, cbi in zip(countera, counterb):
            a = np.array(adj1[cri, :])
            one_hop_neighbor = np.flatnonzero(a)
            b = np.array(adj2[cbi, :])
            # neighbor of counterpart
            new_one_hop_neighbor = np.flatnonzero(b)

            one_hop_neighbor_counter = []

            for count in one_hop_neighbor:
                indx = np.where(count == countera)
                try:
                    one_hop_neighbor_counter.append(counterb[indx[0][0]])
                except Exception:
                    pass

            num_stable_neighbor = np.intersect1d(
                new_one_hop_neighbor, np.array(one_hop_neighbor_counter)).shape[0]
            union_align = np.union1d(new_one_hop_neighbor, np.array(
                one_hop_neighbor_counter)).shape[0]

            sim = float(num_stable_neighbor) / union_align
            mnc += sim

        return mnc / countera.size
    except Exception:
        return -1

def frobenius(pair: GraphPair, P: np.ndarray) -> float:
    """Squared Frobenius distance between the matched adjacency submatrices.

    Lower is better, and requires no ground truth. The value scales with graph
    size, so it is only comparable across alignments of the same pair.

    :param pair: Graph pair (ground truth not required).
    :param P: Similarity matrix of shape [n_src, n_tar].
    """
    ma, mb = scipy.optimize.linear_sum_assignment(-P)
    adj1 = pair.src_adjacency[ma][:, ma]
    adj2 = pair.tar_adjacency[mb][:, mb]
    return float(LA.norm(adj1 - adj2, "fro") ** 2)


def accuracy(pair:GraphPair, p: np.ndarray) -> float:
    # default Hungarian matching, find the best
    # node-to-node assignment.
    ma, mb = scipy.optimize.linear_sum_assignment(-p)

    _, acc, _ = _eval_align(ma, mb, pair.ground_truth[0])
    return  acc

def _eval_align(ma, mb, gmb):
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
    return (gacc, # Global accuracy, unmatched nodes count against the score
            acc, # Accuracy, precision over matched pairs
            alignment) # Node-by-node alignment result table