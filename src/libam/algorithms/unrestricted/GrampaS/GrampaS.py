from dataclasses import dataclass

import numpy as np

from libam.graph.graph_pair import GraphPair
from libam.algorithms.unrestricted.FUGAL.pred import eucledian_dist
from libam.algorithms.utils.feature_util import feature_extraction

from numpy.linalg import eigh
import networkx as nx
from math import floor, log2
import scipy.sparse as sps
import os
import time

from libam.algorithms.algorithm import AlignAlgorithm


def calculate_similarity_scores_from_matrices(G_A, G_B):
    # Step 1: Calculate degrees and normalize
    degrees_A = np.sum(G_A, axis=1)
    degrees_B = np.sum(G_B, axis=1)
    
    sum_degrees_A = np.sum(degrees_A)
    sum_degrees_B = np.sum(degrees_B)
    
    normalized_degrees_A = degrees_A / sum_degrees_A
    normalized_degrees_B = degrees_B / sum_degrees_B
    
    # Step 2: Compute similarity scores
    num_nodes_A = G_A.shape[0]
    num_nodes_B = G_B.shape[0]
    
    similarity_scores = np.zeros((num_nodes_A, num_nodes_B))

    min_degrees = np.minimum.outer(normalized_degrees_A, normalized_degrees_B)
    max_degrees = np.maximum.outer(normalized_degrees_A, normalized_degrees_B)
# Calculate the similarity scores using element-wise division
    similarity_scores = min_degrees / max_degrees
    return similarity_scores


def decompose_laplacian(A):
    # Compute the degree matrix
    D = np.diag(np.sum(A, axis=1))
    print(D)
    n = np.shape(D)[0]

    # Calculate the unnormalized Laplacian matrix
    L = D - A

    # Compute the eigenvalues and eigenvectors of L
    D, V = np.linalg.eigh(L)
    #D, V = seigh(L)
    return [D, V]
def decomposeN_laplacian(A):

    #  adjacency matrix
    start = time.time()
    Deg = np.linalg.inv(np.sqrt(np.diag((np.sum(A, axis=1)))))
    #Deg = np.diag((np.sum(A, axis=1)))

    n = np.shape(Deg)[0]

    #Deg 
    #Deg = sci.linalg.fractional_matrix_power(Deg, -0.5)

    L = np.identity(n) - Deg @ A @ Deg
    end = time.time()
    print("create LApl",end-start)
    start = time.time()
    D, V = np.linalg.eigh(L)
    end = time.time()
    print("Eigen Decomp",end-start)
    return [D, V]

def random_walk_laplacian(A):
    # Calculate the degree matrix D
    D = np.diag(np.sum(A, axis=1))
    #epsilon = 1e-6  # Small constant
    #D_inv = np.linalg.inv(D + epsilon * np.identity(D.shape[0]))
    # Compute the inverse of D
    D_inv = np.linalg.inv(D)
    # Calculate the Random Walk Laplacian L_rw
    L_rw = np.identity(len(A)) - np.dot(D_inv, A)
    #D, V = np.linalg.eigh(L_rw)
    D, V = np.linalg.eig(L_rw)
    return [D, V]
    #return L_rw
def Signless_Laplacian(A):
        # Compute the degree matrix
    D = np.diag(np.sum(A, axis=1))
    n = np.shape(D)[0]
    # Calculate the unnormalized Laplacian matrix
    L = D + A
    # Compute the eigenvalues and eigenvectors of L
    D, V = np.linalg.eig(L)
    return [D, V]

def grampa(Src, Tar, eta):
    """
    Summary or Description of the Function

    Parameters:
    Src (np.array): The nxn adjacency matrix of the first graph
    Tar (np.array): The nxn adjacency matrix of the second graph
    eta (float): The eta value of Eq. 4 in the paper

    Returns:
    Xt similarity Matrix
    """
    n = Src.shape[0]
    l, U = eigh(Src)
    mu, V = eigh(Tar)
    l = np.array([l])
    mu = np.array([mu])

    # Eq.4
    coeff = 1.0 / ((l.T - mu) ** 2 + eta ** 2)
    # Eq. 3
    coeff = coeff * (U.T @ np.ones((n, n)) @ V)
    X = U @ coeff @ V.T

    Xt = X.T
    # Solve with linear assignment maximizing the similarity
    # row,col = linear_sum_assignment(Xt, maximize=True)

    # Alternatively, we can use a more efficient solver.
    # The solver works on cost minimization, so take -X
    # rows, cols = solve_dense(-Xt)
    # return rows, cols
    return Xt


@dataclass
class GrampaS(AlignAlgorithm):
    pair: GraphPair
    eta: float
    initSim: int
    eig_type: int

    @property
    def name(self) -> str:
        return "GrampaS"

    def _align(self):
        # lalpha is unused?
        Src = self.pair.src_adjacency
        Tar = self.pair.tar_adjacency
        eta = self.eta
        initSim = self.initSim
        Eigtype = self.eig_type

        os.environ["MKL_NUM_THREADS"] = "20"

        n = Src.shape[0]
        # Adjancency
        if Eigtype == 0:
            l, U = eigh(Src)
            mu, V = eigh(Tar)

        elif Eigtype == 1:  # Laplacian
            l, U = decompose_laplacian(Src)
            mu, V = decompose_laplacian(Tar)

        elif Eigtype == 2:  # RandomWalk Laplacian
            l, U = random_walk_laplacian(Src)
            mu, V = random_walk_laplacian(Tar)
        elif Eigtype == 3:  # Singless Laplacian
            l, U = Signless_Laplacian(Src)
            mu, V = Signless_Laplacian(Tar)

        else:  # Normalized Laplacian
            l, U = decomposeN_laplacian(Src)
            mu, V = decomposeN_laplacian(Tar)

        l = np.array([l])
        mu = np.array([mu])
        dtype = np.float32
        # Eq.4

        coeff = 1 / ((l.T - mu) ** 2 + eta ** 2)

        if initSim == 1:
            Src1 = nx.from_numpy_array(Src)
            Tar1 = nx.from_numpy_array(Tar)
            F1 = feature_extraction(Src1, True)
            F2 = feature_extraction(Tar1, True)
            K = eucledian_dist(F1, F2, n)
            L = np.max(K) - K
            coeff = coeff * (U.T @ L @ V)
        else:
            coeff = coeff * (U.T @ np.ones((n, n)) @ V)
        X = U @ coeff @ V.T
        Xt = X
        return Xt