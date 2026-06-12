from dataclasses import dataclass

import scipy

from libam.graph.graph_pair import GraphPair
from .model.GromovWassersteinLearning import GromovWassersteinLearning
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import numpy as np
import scipy.sparse as sps

from libam.algorithms.algorithm import AlignAlgorithm


#original code from https://github.com/HongtengXu/gwl

@dataclass
class GWL(AlignAlgorithm):
    pair: GraphPair
    epochs: int
    batch_size: int
    use_cuda: bool
    strategy: str
    beta: float
    outer_iteration: int
    inner_iteration: int
    sgd_iteration: int
    display: bool
    prefix: str
    prior: bool
    # Hyperparameters
    dimension: int
    loss_type: str
    cost_type: str
    ot_method: str
    # Scheduler / hardware
    lr: float
    gamma: float
    max_cpu: int

    def _align(self) -> np.ndarray | torch.Tensor | scipy.sparse.csr_matrix:
        Src = self.pair.src_adjacency
        Tar = self.pair.tar_adjacency
        Se = np.array(sps.find(sps.csr_matrix(Src))[:2]).T
        Te = np.array(sps.find(sps.csr_matrix(Tar))[:2]).T

        data = {
            'src_index': {float(i): i for i in range(np.amax(Se) + 1)},
            'src_interactions': Se.tolist(),
            'tar_index': {float(i): i for i in range(np.amax(Te) + 1)},
            'tar_interactions': Te.tolist(),
            'mutual_interactions': None,
        }

        hyperpara = {
            'src_number': len(data['src_index']),
            'tar_number': len(data['tar_index']),
            'dimension': self.dimension,
            'loss_type': self.loss_type,
            'cost_type': self.cost_type,
            'ot_method': self.ot_method,
        }

        opt = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'use_cuda': self.use_cuda,
            'strategy': self.strategy,
            'beta': self.beta,
            'outer_iteration': self.outer_iteration,
            'inner_iteration': self.inner_iteration,
            'sgd_iteration': self.sgd_iteration,
            'display': self.display,
            'prefix': self.prefix,
            'prior': self.prior,
        }

        if self.max_cpu > 0:
            torch.set_num_threads(self.max_cpu)

        gwd_model = GromovWassersteinLearning(hyperpara)

        # initialize optimizer
        optimizer = optim.Adam(gwd_model.gwl_model.parameters(), lr=self.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma) if self.gamma else None

        gwd_model.train_without_prior(data, optimizer, opt, scheduler=scheduler)
        return gwd_model.trans
