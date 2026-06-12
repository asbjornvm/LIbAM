import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy
import torch
from torch_geometric.utils import add_self_loops, to_dense_adj

from libam.algorithms.restricted.SlotA.GWLTorch import gw_torch
from libam.algorithms.restricted.SlotA.utils import euclidean_proj_simplex
from libam.graph.graph_pair import GraphPair
from libam.algorithms.algorithm import AlignAlgorithm


def _norm_aggr(x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Symmetric normalized aggregation: D^{-1/2} A D^{-1/2} x (no learnable weights)."""
    row, col = edge_index
    ones = torch.ones(row.size(0), device=x.device)
    deg = torch.zeros(num_nodes, device=x.device).scatter_add_(0, row, ones)
    deg_inv_sqrt = deg.pow(-0.5).clamp(min=0)
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
    return adj @ x


@dataclass
class SlotA(AlignAlgorithm):
    pair: GraphPair
    step_size: float
    bases: int
    joint_epoch: int
    epoch: int
    gw_beta: float

    def __post_init__(self) -> None:
        self.__name__: str = "SLOTA"

    def _align(self) -> np.ndarray | torch.Tensor | scipy.sparse.csr_matrix:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Aadj = nx.to_numpy_array(self.pair.src)
        Badj = nx.to_numpy_array(self.pair.tar)

        Afeat = self.pair.src_features if self.pair.src_features is not None else np.ones((self.pair.src.number_of_nodes(), 1))
        Bfeat = self.pair.tar_features if self.pair.tar_features is not None else np.ones((self.pair.tar.number_of_nodes(), 1))

        Adim, Bdim = Afeat.shape[0], Bfeat.shape[0]

        Ag = torch.tensor(np.array(np.nonzero(Aadj)), dtype=torch.long)
        Bg = torch.tensor(np.array(np.nonzero(Badj)), dtype=torch.long)

        Afeat -= Afeat.mean(0)
        Bfeat -= Bfeat.mean(0)

        Afeat = torch.tensor(Afeat).float().to(device)
        Bfeat = torch.tensor(Bfeat).float().to(device)

        layers = self.bases - 2
        Afeats = [Afeat]
        Bfeats = [Bfeat]
        Ag = Ag.to(device)
        Bg = Bg.to(device)
        Aei_sl = add_self_loops(Ag, num_nodes=Adim)[0]
        Bei_sl = add_self_loops(Bg, num_nodes=Bdim)[0]
        for i in range(layers):
            Afeats.append(_norm_aggr(Afeats[-1], Aei_sl, Adim).detach().clone())
            Bfeats.append(_norm_aggr(Bfeats[-1], Bei_sl, Bdim).detach().clone())

        Asims = [to_dense_adj(Ag, max_num_nodes=Adim).squeeze(0)]
        Bsims = [to_dense_adj(Bg, max_num_nodes=Bdim).squeeze(0)]
        for i in range(len(Afeats)):
            Afeat = Afeats[i]
            Bfeat = Bfeats[i]
            Afeat = Afeat / (Afeat.norm(dim=1)[:, None] + 1e-16)
            Bfeat = Bfeat / (Bfeat.norm(dim=1)[:, None] + 1e-16)
            Asim = Afeat.mm(Afeat.T)
            Bsim = Bfeat.mm(Bfeat.T)
            Asims.append(Asim)
            Bsims.append(Bsim)

        Adim, Bdim = Afeat.shape[0], Bfeat.shape[0]
        a = torch.ones([Adim, 1]).to(device) / Adim
        b = torch.ones([Bdim, 1]).to(device) / Bdim
        X = a @ b.T
        As = torch.stack(Asims, dim=2)
        Bs = torch.stack(Bsims, dim=2)

        alpha0 = np.ones(layers + 2).astype(np.float32) / (layers + 2)
        beta0 = np.ones(layers + 2).astype(np.float32) / (layers + 2)
        for ii in range(self.joint_epoch):
            alpha = torch.autograd.Variable(torch.tensor(alpha0)).to(device)
            alpha.requires_grad = True
            beta = torch.autograd.Variable(torch.tensor(beta0)).to(device)
            beta.requires_grad = True
            A = (As * alpha).sum(2)
            B = (Bs * beta).sum(2)
            objective = (A ** 2).mean() + (B ** 2).mean() - torch.trace(A @ X @ B @ X.T)
            alpha_grad = torch.autograd.grad(outputs=objective, inputs=alpha, retain_graph=True)[0]
            alpha = alpha - self.step_size * alpha_grad
            alpha0 = alpha.detach().cpu().numpy()
            alpha0 = euclidean_proj_simplex(alpha0)
            beta_grad = torch.autograd.grad(outputs=objective, inputs=beta)[0]
            beta = beta - self.step_size * beta_grad
            beta0 = beta.detach().cpu().numpy()
            beta0 = euclidean_proj_simplex(beta0)
            X = gw_torch(A.clone().detach(), B.clone().detach(), a, b, X.clone().detach(), beta=self.gw_beta,
                         outer_iter=1, inner_iter=50).clone().detach()
            if ii == self.joint_epoch - 1:
                logging.getLogger(__name__).debug(f"{alpha0}, {beta0}")
                X = gw_torch(A.clone().detach(), B.clone().detach(), a, b, X, beta=self.gw_beta,
                             outer_iter=self.epoch - self.joint_epoch, inner_iter=20, gt=None)
        return X.T

