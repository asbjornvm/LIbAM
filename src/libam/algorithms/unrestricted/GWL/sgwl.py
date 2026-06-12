# from .model.GromovWassersteinLearning import GromovWassersteinLearning
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import torch
#original code https://github.com/HongtengXu/s-gwl
import numpy as np
# import scipy.sparse as sps
from .methods import DataIO, GromovWassersteinGraphToolkit as GwGt
import networkx as nx
import torch

from libam.algorithms.algorithm import AlignAlgorithm
from libam.graph.graph_pair import GraphPair

# methods = ['gwl', 's-gwl-3', 's-gwl-2', 's-gwl-1']
cluster_num = [2, 4, 8]
partition_level = [3, 2, 1]


class SGWL(AlignAlgorithm):
    def __init__(
        self,
        pair: GraphPair,
        mn: int,
        loss_type: str = "L2",
        ot_method: str = "proximal",
        beta: float = 0.2,
        iter_bound: float = 1e-10,
        inner_iteration: int = 2,
        sk_bound: float = 1e-30,
        node_prior: float = 1000,
        max_iter: int = 4,
        cost_bound: float = 1e-26,
        update_p: bool = False,
        lr: float = 0,
        alpha: float = 1,
        max_cpu: int = 0,
        clus: int = 2,
        level: int = 3,
    ):
        self.pair = pair
        self.mn = mn
        # Gromov-Wasserstein optimal-transport hyperparameters
        self.loss_type = loss_type
        self.ot_method = ot_method
        self.beta = beta
        self.iter_bound = iter_bound
        self.inner_iteration = inner_iteration
        self.sk_bound = sk_bound
        self.node_prior = node_prior
        self.max_iter = max_iter
        self.cost_bound = cost_bound
        self.update_p = update_p
        self.lr = lr
        self.alpha = alpha
        self.clus = clus
        self.level = level

    def _align(self):
        Src = self.pair.src_adjacency
        Tar = self.pair.target_adjacency

        for i in range(Src.shape[0]):
            row_sum1 = np.sum(Src[i, :])
        # If the sum of the row is zero, add a self-loop
            if row_sum1 == 0:
                Src[i, i] = 1
        for i in range(Tar.shape[0]):
            row_sum = np.sum(Tar[i, :])

        # If the sum of the row is zero, add a self-loop
            if row_sum == 0:
                Tar[i, i] = 1
        # print(Src.tolist())
        p_s, cost_s, idx2node_s = DataIO.extract_graph_info(
            nx.Graph(Src), weights=None)
        # print(cost_s.A.tolist())
        p_s /= np.sum(p_s)
        p_t, cost_t, idx2node_t = DataIO.extract_graph_info(
            nx.Graph(Tar), weights=None)
        p_t /= np.sum(p_t)

        ot_dict = {
            'loss_type': self.loss_type,
            'ot_method': self.ot_method,
            'beta': self.beta,
            'outer_iteration': Src.shape[0],
            'iter_bound': self.iter_bound,
            'inner_iteration': self.inner_iteration,
            'sk_bound': self.sk_bound,
            'node_prior': self.node_prior,
            'max_iter': self.max_iter,
            'cost_bound': self.cost_bound,
            'update_p': self.update_p,
            'lr': self.lr,
            'alpha': self.alpha,
        }

        if self.mn == 0:
            pairs_idx, pairs_name, pairs_confidence, trans = GwGt.direct_graph_matching(
                0.5 * (cost_s + cost_s.T), 0.5 * (cost_t + cost_t.T), p_s, p_t, idx2node_s, idx2node_t, ot_dict)
        else:
            pairs_idx, pairs_name, pairs_confidence, trans = GwGt.recursive_direct_graph_matching(
                0.5 * (cost_s + cost_s.T), 0.5 * (cost_t + cost_t.T), p_s, p_t,
                idx2node_s, idx2node_t, ot_dict, weights=None, predefine_barycenter=False,
                cluster_num=self.clus, partition_level=self.level, max_node_num=200
            )
        pairs = np.array(pairs_name)[::-1].T

        # return res
        return trans
