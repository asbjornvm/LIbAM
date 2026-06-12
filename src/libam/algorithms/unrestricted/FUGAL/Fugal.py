#Fugal Algorithm was provided by anonymous authors.
from dataclasses import dataclass, field
import networkx as nx
import numpy as np
import scipy
import torch

from libam.graph.graph_pair import GraphPair
from libam.algorithms.algorithm import AlignAlgorithm
from .pred import Degree_Features, convex_init, eucledian_dist
from libam.algorithms.utils.feature_util import feature_extraction

@dataclass
class Fugal(AlignAlgorithm):
    pair: GraphPair
    iterations: int
    simple: bool
    mu: float
    efn: int = 5

    def __post_init__(self) -> None:
        self.__name__: str = "FUGAL"

    def _align(self) -> np.ndarray | torch.Tensor | scipy.sparse.csr_matrix:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        src: np.ndarray = self.pair.src_adjacency
        tar: np.ndarray = self.pair.tar_adjacency
        for i in range(src.shape[0]):
            row_sum1: float = np.sum(src[i, :])

        # If the sum of the row is zero, add a self-loop
            if row_sum1 == 0:
                src[i, i] = 1
        for i in range(tar.shape[0]):
            row_sum: float = np.sum(tar[i, :])

            # If the sum of the row is zero, add a self-loop
            if row_sum == 0:
                tar[i, i] = 1
        n1 = tar.shape[0]
        n2 = src.shape[0]
        n = max(n1, n2)
        src1 = nx.from_numpy_array(src)
        tar1 = nx.from_numpy_array(tar)
        a = torch.tensor(src, dtype=torch.float64).to(device)
        b = torch.tensor(tar, dtype=torch.float64).to(device)

        f1, f2 = None, None
        if self.efn < 5:
            f1 = Degree_Features(src1, self.efn) * n1
            f2 = Degree_Features(tar1, self.efn) * n1
        # EFN 5 equals fugal
        if self.efn == 5:
            f1 = feature_extraction(src1, self.simple)
            f2 = feature_extraction(tar1, self.simple)
        d = eucledian_dist(f1, f2, n)
        d = torch.tensor(d, dtype=torch.float64).to(device)

        p = convex_init(a, b, d, self.mu, self.iterations)
        p = p.detach().cpu().numpy()
        return p
