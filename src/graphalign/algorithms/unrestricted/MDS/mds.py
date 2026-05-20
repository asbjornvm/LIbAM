from dataclasses import dataclass

from graphalign import GraphPair
from graphalign.algorithms.algorithm import Algorithm
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from graphalign.algorithms.unrestricted.MDS.joint_mds import JointMDS
from scipy.sparse import csr_matrix


def get_quadratic_inverse_weight(shortest_path):
    w = 1.0 / shortest_path**4
    w[np.isinf(w)] = 0.0
    w /= w.sum()
    return w

def compute_shortest_path(adj):
    adj.data = 1.0 / (1.0 + adj.data)
    #adj=1.0/(1.0+adj)
    # adj.data = 1. - adj.data
    adj = dijkstra(csgraph=adj, directed=False, return_predecessors=False)
    adj /= adj.mean()
    return adj

def normalize_adj(adj):
    degree = np.asarray(adj.sum(1))
    d_inv_sqrt = np.power(degree, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)  # .tocoo()

@dataclass
class Mds(Algorithm):
    pair: GraphPair
    n_components: int
    alpha: float
    max_iter: int
    eps: float
    tol: float
    min_eps: float
    eps_annealing: bool
    alpha_annealing: bool
    gw_init: bool
    return_stress: bool

    @property
    def name(self):
        return "MDS"

    def evaluate(self) -> np.ndarray:
        A = self.pair.src_adjacency
        B = self.pair.tar_adjacency

        A = csr_matrix(A)
        B = csr_matrix(B)
        adj_s_normalized = normalize_adj(A)
        adj_t_normalized = normalize_adj(B)
        adj_s_normalized = compute_shortest_path(adj_s_normalized)
        adj_t_normalized = compute_shortest_path(adj_t_normalized)
        w1 = get_quadratic_inverse_weight(adj_s_normalized)
        w2 = get_quadratic_inverse_weight(adj_t_normalized)
        w1 = torch.from_numpy(w1)
        w2 = torch.from_numpy(w2)
        torch.manual_seed(1)
        JMDS = JointMDS(
            n_components=self.n_components,
            alpha=self.alpha,
            #alpha=0.1,
            max_iter=self.max_iter,
            eps=self.eps,
            #eps=1,
            tol=self.tol,
            min_eps=self.min_eps,
            eps_annealing=self.eps_annealing,
            alpha_annealing=self.alpha_annealing,
            gw_init=self.gw_init,
            return_stress=self.return_stress
        )
        Z1, Z2, P = JMDS.fit_transform(
            torch.from_numpy(adj_s_normalized),
            torch.from_numpy(adj_t_normalized),
            w1=w1,
            w2=w2#,
            #a=torch.from_numpy(weight_s),
            #b=torch.from_numpy(weight_t),
        )
        cost_matrix = P.numpy()
        return cost_matrix