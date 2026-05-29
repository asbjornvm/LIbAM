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

from graphalign.algorithms.algorithm import Algorithm
from graphalign.graph import GraphPair

# methods = ['gwl', 's-gwl-3', 's-gwl-2', 's-gwl-1']
cluster_num = [2, 4, 8]
partition_level = [3, 2, 1]


class SGWL(Algorithm):
    def __init__(self, pair: GraphPair, ot_dict: dict, mn: int, max_cpu: int = 0, clus: int = 2, level: int = 3):
        self.pair = pair
        self.ot_dict = ot_dict
        self.mn = mn
        self.max_cpu = max_cpu
        self.clus = clus
        self.level = level

    def _evaluate(self):
        print("SGWL")
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
        if self.max_cpu > 0:
            torch.set_num_threads(self.max_cpu)

        ot_dict = {
            **self.ot_dict,
            'outer_iteration': Src.shape[0]
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
