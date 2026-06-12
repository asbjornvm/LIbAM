"""Implementation of Alpine algorithm."""
from abc import ABC

from .anchor_synchronize import seed_link, synchronize_features  # type: ignore
import numpy as np
import networkx as nx
import os
from .perm_projection import Hungarian_all
from .sinkhornP import SinkhornKnopp
from sklearn.metrics.pairwise import euclidean_distances        # type: ignore
from sklearn.preprocessing import StandardScaler                # type: ignore
from .structural_features import structural_features            # type: ignore
import torch
from tqdm import tqdm
import warnings
from typing import Any, Literal, Union
from dataclasses import dataclass

from libam.graph.graph_pair import GraphPair
from libam.algorithms.algorithm import AlignAlgorithm

warnings.filterwarnings('ignore')
os.environ["MKL_NUM_THREADS"] = "40"
torch.set_num_threads(40)


def _gradient(A: torch.Tensor, B: torch.Tensor, P: torch.Tensor,
              Kstr: Union[torch.Tensor , None] = None,
              Kattr: Union[torch.Tensor, None] = None,
              Ip: Union[torch.Tensor , None] = None,
              t: int = 0,
              A0: Union[float , None] = None,
              dd: float = 1.0):
    gradP = t*(1.0-2.0*P)
    if Ip is not None:
        gradP += (-4.0*Ip.T@(A-Ip@P@B@P.T@Ip.T)@Ip@P@B)*dd
    else:
        gradP += (-4.0*(A-P@B@P.T)@P@B)*dd
    if Kstr is not None:
        gradP += Kstr
    if Kattr is not None and A0 is not None:
        gradP += (gradP.abs().mean()/A0)*Kattr
    return gradP


def _alpine_run(A: torch.Tensor, B: torch.Tensor,
                Kstru: Union[torch.Tensor , None] = None,
                Kattr: Union[torch.Tensor , None] = None,
                niter: int = 10,
                sinkhorn_iters: int = 500,
                sinkhorn_tol: float = 1e-5,
                anchor_pairs=None,
                method: Literal['FW', 'AMD'] = 'FW'):
    assert len(A) <= len(B), "A should not be larger than B"
    # Number of nodes
    n1, n2 = len(A), len(B)
    n_anchor = 0 if anchor_pairs is None else len(anchor_pairs)
    n1_ = n1
    # Init matrices
    Ip: torch.Tensor | None = None
    if n1 < n2:
        n1_ = n1 + 1
        Ip = torch.zeros((n1, n1_), dtype=torch.float64)
        Ip.fill_diagonal_(1.0)
    P = torch.zeros((n1_, n2), dtype=torch.float64)
    P[:n1, :n2] = 1.0 / (n2 - n_anchor)
    if n1 < n2:
        P[-1, :] = (n2-n1)/(n2-n_anchor)
    if anchor_pairs is not None:
        for i, j in anchor_pairs:
            P[i, :] = 0
            P[:, j] = 0
            P[i, j] = 1
            if Kstru is not None:
                Kstru[i, j] = 0
    Plast = P[-1, :]
    Q = 1.0*P
    # Init vectors
    sumvec1 = torch.ones(n1_, dtype=torch.float64)
    sumvec2 = torch.ones(n2, dtype=torch.float64)
    if n1 < n2:
        sumvec1[-1] = n2-n1
    # Init scalars
    A0: None | float = None
    if Kattr is not None:
        A0 = float(Kattr.abs().mean()) + 1e-4
    dd = 2.0 if ((A.sum()/n1 < 3) or (B.sum()/n2 < 3)) else 1.0
    
    pbar = tqdm(total=niter * 10, desc="Total Progress")
    # Increasing concavity
    for t in range(niter):
        # Frank-Wolfe iteration
        for it in range(1, 11):
            pbar.update(1)
            # Update 1
            alpha = 2.0/(it+2.0) if method == 'FW' else 4.0/(it+4.0)
            eta = 1.0 if method == 'FW' else 2.0/(it+2.0)
            P_hat = P if method == 'FW' else (1-alpha)*P + alpha*Q
            # Gradient computation
            gradP = _gradient(A, B, P_hat, Kstru, Kattr, Ip, t, A0, dd)
            # Projection
            M = torch.exp(
                -eta*gradP) if method == 'FW' else Q*torch.exp(-eta*gradP)
            Q = SinkhornKnopp(M, sumvec1, sumvec2,
                              sinkhorn_iters, sinkhorn_tol) 
            #Q=ot.sinkhorn(sumvec1, sumvec2,gradP, 
            #               1, numItermax = 3000, stopThr = 1e-12,warn=True)
            if n1 < n2:
                Q[-1, :] = Plast
            if anchor_pairs is not None:
                for i, j in anchor_pairs:
                    Q[i, :] = 0
                    Q[:, j] = 0
                    Q[i, j] = 1
            # Update 2
            P = (1-alpha)*P + alpha * Q
    return P[:n1, :n2]


def _init_alpine(G1: nx.Graph, G2: nx.Graph,
                 attr1: Union[np.ndarray , None] = None,
                 attr2: Union[np.ndarray , None] = None,
                 anchor_pairs: Union[np.ndarray , None] = None,
                 k: Union[int , None] = None,
                 connected: bool = False,
                 mu: Union[float , None] = None,
                 method: Literal['FW', 'AMD'] = 'FW'):
    # Checking input
    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()
    assert n2 >= n1, "Nodes of G2 should at least the number of nodes in G1"
    assert not ((attr1 is None) ^ (attr2 is None)), "Missing attributes"
    if attr1 is not None and attr2 is not None:
        assert len(attr1) == n1, "Size of G1 and attr1 do not match"
        assert len(attr2) == n2, "Size of G2 and attr2 do not match"
    if k is not None:
        assert k <= min(n1, n2), "k should be smaller than min(n1, n2)"
    if connected:
        ccG1 = nx.connected_components(G1)
        ccG2 = nx.connected_components(G2)
        kk = k if k is not None else n1
        assert max(len(c1) for c1 in ccG1) >= kk, f'Max size of cc in G1<{kk}'
        assert max(len(c2) for c2 in ccG2) >= kk, f'Max size of cc in G2<{kk}'
    if mu is not None:
        assert mu >= 0, "Coeff. mu should be non-negative."
    assert method in ['FW', 'AMD'], "Available methods ['FW', 'AMD']."
    # Copy and update graphs
    _G1 = G1.copy()
    _G2 = G2.copy()
    has_anchors = anchor_pairs is not None and len(anchor_pairs) > 0
    if has_anchors:
        seed_link(_G1, _G2, anchor_pairs)
    for node in nx.isolates(_G1):
        _G1.add_edge(node, node)
    for node in nx.isolates(_G2):
        _G2.add_edge(node, node)
    use_aug = False
    if n1 < n2:
        use_aug = True
        _G1.add_node(n1)
        _G1.add_edge(n1, n1)
    # Distance matrices
    Kstr, Kattr = None, None
    if mu is None and (attr1 is None or attr2 is None):
        mu = 1.0
    if mu is not None:
        Kstr = mu*torch.tensor(
               euclidean_distances(structural_features(_G1),
                                   structural_features(_G2)),
               dtype=torch.float64)
    if attr1 is not None and attr2 is not None:
        _attr1, _attr2 = attr1, attr2
        if has_anchors:
            _attr1, _attr2 = synchronize_features(attr1, attr2, anchor_pairs)
        scaler = StandardScaler()
        _attr1 = scaler.fit_transform(_attr1)
        _attr2 = scaler.transform(_attr2)
        Kattr = torch.tensor(euclidean_distances(_attr1, _attr2),
                             dtype=torch.float64)
        if use_aug:
            Kattr = torch.vstack([Kattr, torch.zeros((1, Kattr.shape[1]))])
    # Adjacency matrices. Force canonical node ordering (0..n-1) to not scramble anchor data
    A = torch.tensor(nx.to_numpy_array(_G1, nodelist=range(_G1.number_of_nodes())), dtype=torch.float64)
    B = torch.tensor(nx.to_numpy_array(_G2, nodelist=range(_G2.number_of_nodes())), dtype=torch.float64)
    A = A[:n1, :n1]
    A.fill_diagonal_(0)
    B.fill_diagonal_(0)
    return A, B, Kstr, Kattr


@dataclass
class Alpine(AlignAlgorithm):
    pair: GraphPair
    anchor_links: Union[np.ndarray , None] = None
    k: Union[int , None] = None
    connected: bool = False
    mu: Union[float , None] = None
    niter: int = 10
    sinkhorn_iters: int = 500
    sinkhorn_tol: float = 1e-9
    method: Literal['FW', 'AMD'] = 'FW'
    """
    Implementation of Alpine algorithm.\n
    This algorithm applies to all GA scenarios:
    (i) without attributes or anchor pairs,
    (ii) with attributes, and
    (iii) with anchor pairs.

    Parameters
    ----------
        G1 (nx.Graph): Source graph G1.
        G2 (nx.Graph): Target graph G2.
        attr1 (np.ndarray | None, optional): Node-attributes of G1.
            Defaults to None.
        attr2 (np.ndarray | None, optional): Node-attributes of G2.
            Defaults to None.
        anchor_links (List[Tuple[int, int]] | None, optional): Anchor-pairs.
            Defaults to None.
        k (int | None, optional): Size of alignment. If None then k=n1.
            Defaults to None.
        connected (bool, optional): Whether or not the selected nodes
                                    in bijection should form a connected
                                    subraph in G1 and G2.
            Defaults to False.
        mu (float | None, optional): Coefficient of structural-distance matrix.
            Defaults to None. If no attributes is set to 1.
        niter (int, optional): Number of (out-)iterations. Defaults to 10.
        sinkhorn_iters (int, optional): Number of SinkhornKnopp iterations.
            Defaults to 500.
        sinkhorn_tol (float, optional): SinkhornKnopp tolerance.
            Defaults to 1e-9.
        method (Literal['FW', 'AMD'], optional): The optimization algorithm to
                                                 be used; 'FW' for Frank-Wolfe,
                                                 'AMD' for Accelerated Mirror
                                                 Descent.
            Defaults to 'FW'.

    Returns
    -------
    ret_dict : dict[str, Any]
            Dictionary containing: \\
            i) The bijection map from
            nodes of G1 (row_ind) to nodes of G2 (col_ind). \\
            ii) The QAP cost of edge-disagreements (QAP),\\
            i.e. ||A-PBP.T||^2-(2m1-(vec_A.T)A(vec_A)),
            where m1 is the number of edges in A.\\
            When k=n1 (default), then QAP cost equals to |A-PBP.T||^2.\\
            iii) If applied, the LAP costs of structural distances (LAP_stru)
            and attributes distances (LAP_attr),\\
            i.e. tr(P.T Kstru) and tr(P.T Kattr), resp.
    """

    @property
    def name(self):
        return "ALPINE"

    def _align(self) -> np.ndarray:
        # Init parameters and check input
        A, B, Kstru, Kattr = _init_alpine(self.pair.src, self.pair.tar, self.pair.src_features, self.pair.tar_features,
                                          self.anchor_links, self.k, self.connected,
                                          self.mu, self.method)
        # Solve objective
        P = _alpine_run(A, B, Kstru, Kattr,
                        self.niter, self.sinkhorn_iters, self.sinkhorn_tol,
                        self.anchor_links, self.method)

        # Prompt projection to Partial Permutation Matrix
        P, row_ind, col_ind = Hungarian_all(P, self.k, self.connected, self.pair.src, self.pair.tar, True)

        # Diagnostic metrics, stored on the instance, can be used in the future to debug.
        vec_A = torch.sum(P, 1).reshape(-1, 1)
        self.metrics: dict[str, Any] = {'row_ind': row_ind, 'col_ind': col_ind}
        self.metrics['QAP'] = torch.norm(A - P@B@P.T)**2 - (
            2*self.pair.src.number_of_edges()-(vec_A.T@A@vec_A).item())
        if Kstru is not None:
            self.metrics['LAP_stru'] = torch.trace(P.T@Kstru[:len(A), :len(B)])
        if Kattr is not None:
            self.metrics['LAP_attr'] = torch.trace(P.T@Kattr[:len(A), :len(B)])

        return P
