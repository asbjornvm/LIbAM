import logging
from typing import Any, Union, List, Tuple, Literal

import numpy as np
import torch

from libam.algorithms.unrestricted.FUGAL.Fugal import Fugal as _Fugal
from libam.algorithms.unrestricted.GrampaS.GrampaS import GrampaS as _GrampaS
from libam.algorithms.unrestricted.Grampa.Grampa import Grampa as _Grampa
from libam.algorithms.unrestricted.GWL.gwl import GWL as _GWL
from libam.algorithms.unrestricted.GWL.sgwl import SGWL as _SGWL
from libam.algorithms.unrestricted.CONE.conealign import Cone as _Cone
from libam.algorithms.unrestricted.NSD.NSD import NSD as _NSD
from libam.algorithms.unrestricted.Path.path import Path as _Path
from libam.algorithms.unrestricted.MMNC.mmnc import Mmnc as _Mmnc
from libam.algorithms.unrestricted.DSPP.Dspp import DSPP as _DSPP
from libam.algorithms.unrestricted.GrASp.grasp import Grasp as _Grasp
from libam.algorithms.unrestricted.KLAUS.klaus import Klaus as _Klaus
from libam.algorithms.unrestricted.LREA.lrea import Lrea as _Lrea
from libam.algorithms.unrestricted.MDS.mds import Mds as _Mds
from libam.algorithms.unrestricted.NetAlign.netalign import NetAlign as _NetAlign
from libam.algorithms.unrestricted.REGAL.regal import Regal as _Regal
from libam.algorithms.unrestricted.PARROT.Parrot import Parrot as _Parrot
from libam.algorithms.unrestricted.isorank.isorank import Isorank as _Isorank
from libam.algorithms.unrestricted.Got.got import Got as _Got
from libam.algorithms.unrestricted.Fgot.fgot import Fgot as _FGot
from libam.algorithms.restricted.GradP.gradp import GradP as _GradP
from libam.algorithms.restricted.JOENA.joena import Joena as _Joena
from libam.algorithms.restricted.HTC.htc import HTC as _Htc
from libam.algorithms.restricted.SlotA.slot_a import SlotA as _SlotA
from libam.algorithms.restricted.ALPINE.alpine import Alpine as _ALPINE
#from libam.algorithms.NotImplemented.NextAlign.NextA import NextAlign as _NextAlign

from libam.graph.graph_pair import GraphPair as _GraphPair
from libam.algorithms.algorithm import AlignAlgorithm

logger = logging.getLogger(__name__)

def _warn_unused(name:str, **kwargs) -> None:
    for key in kwargs:
        logger.warning(f"{name} received unknown kwarg '{key}' - it will be ignored")


def fugal(pair: _GraphPair, iterations: int = 1, simple: bool = True, mu: float = 0.05, efn: int = 3, **kwargs: Any) -> _Fugal:
    """
    Feature-fortified Unrestricted Graph Alignment.

    :param pair: Graph Pair object
    :param iterations: Number of iterations for the alignment
    :param simple: Whether to use simple feature extraction
    :param mu: Regularisation parameter
    :param efn: Edge feature number (default 5 = standard FUGAL)
    """
    alg = _Fugal(pair, iterations=iterations, simple=simple, mu=mu, efn=efn)
    _warn_unused(alg.name, **kwargs)
    return alg


def grampa_s(pair: _GraphPair, eta: float = 0.2, init_sim: int = 1, eig_type: int = 0, **kwargs: Any) -> _GrampaS:
    """
    GRAMPA-S

    :param pair: Graph Pair object
    :param eta: Regularisation parameter
    :param init_sim: Initial similarity type
    :param eig_type: Eigenvalue type
    """
    alg = _GrampaS(pair, eta=eta, initSim=init_sim, eig_type=eig_type)
    _warn_unused(alg.name, **kwargs)
    return alg


def grampa(pair: _GraphPair, eta: float = 0.2, **kwargs: Any) -> _Grampa:
    """
    GRAMPA

    :param pair: Graph Pair object
    :param eta: Regularisation parameter
    """
    alg = _Grampa(pair, eta=eta)
    _warn_unused(alg.name, **kwargs)
    return alg


def cone(
    pair: _GraphPair,
    dim: int = 128,
    window: int = 10,
    negative: float = 1.0,
    niter_init: int = 10,
    reg_init: float = 1.0,
    lr: float = 1.0,
    bsz: int = 10,
    nepoch: int = 5,
    niter_align: int = 10,
    reg_align: float = 0.05,
    embsim: str = "euclidean",
    numtop: int = 10,
    **kwargs: Any,
) -> _Cone:
    """
    CONE (Consistent Network Alignment with Proximity-Preserving Node Embedding)

    :param pair: Graph Pair object
    :param dim: Embedding dimension
    :param window: Context window size
    :param negative:
    :param niter_init:
    :param reg_init:
    :param lr: Learning rate
    :param bsz: Batch size
    :param nepoch: Number of epochs
    :param niter_align:
    :param reg_align:
    :param embsim: Embedding similarity metric
    :param numtop: Number of top candidates
    """
    alg = _Cone(pair, dim=dim, window=window, negative=negative, niter_init=niter_init,
                reg_init=reg_init, lr=lr, bsz=bsz, nepoch=nepoch, niter_align=niter_align,
                reg_align=reg_align, embsim=embsim, numtop=numtop)
    _warn_unused(alg.name, **kwargs)
    return alg


def gwl(
    pair: _GraphPair,
    # Optimiser options
    epochs: int = 1,
    batch_size: int = 64,
    use_cuda: bool = False,
    strategy: str = 'hard',
    beta: float = 0.1,
    outer_iteration: int = 20,
    inner_iteration: int = 10,
    sgd_iteration: int = 200,
    display: bool = False,
    prefix: str = '',
    prior: bool = False,
    # Hyperparameters
    dimension: int = 64,
    loss_type: str = 'L2',
    cost_type: str = 'cosine',
    ot_method: str = 'proximal',
    # Scheduler / hardware
    lr: float = 0.01,
    gamma: float = 0.5,
    max_cpu: int = 0,
    **kwargs: Any,
) -> _GWL:
    """
    GWL (Gromov-Wasserstein Learning)

    :param pair: Graph Pair object
    :param epochs: Training epochs
    :param batch_size: Batch size
    :param use_cuda: Use CUDA if available
    :param strategy: Sampling strategy, 'hard' or 'soft'
    :param beta:
    :param outer_iteration:
    :param inner_iteration:
    :param sgd_iteration:
    :param display: Print training progress
    :param prefix:
    :param prior:
    :param dimension:
    :param loss_type:
    :param cost_type: Cost metric type
    :param ot_method: Optimal transport solver method
    :param lr: Learning rate
    :param gamma:
    :param max_cpu: Max CPU threads - 0 for unlimited
    """
    alg = _GWL(
        pair, epochs=epochs, batch_size=batch_size, use_cuda=use_cuda, strategy=strategy,
        beta=beta, outer_iteration=outer_iteration, inner_iteration=inner_iteration,
        sgd_iteration=sgd_iteration, display=display, prefix=prefix, prior=prior,
        dimension=dimension, loss_type=loss_type, cost_type=cost_type, ot_method=ot_method,
        lr=lr, gamma=gamma, max_cpu=max_cpu,
    )
    _warn_unused(alg.name, **kwargs)
    return alg


def sgwl(
    pair: _GraphPair,
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
    clus: int = 2,
    level: int = 3,
    **kwargs: Any,
) -> _SGWL:
    """
    SGWL (Scalable Gromov-Wasserstein Learning)

    :param pair: Graph Pair object
    :param mn: 
    :param loss_type:
    :param ot_method:
    :param beta: 
    :param iter_bound: 
    :param inner_iteration: 
    :param sk_bound:
    :param node_prior: 
    :param max_iter: 
    :param cost_bound: 
    :param update_p:
    :param lr:
    :param alpha:
    :param clus:
    :param level:
    """
    alg = _SGWL(
        pair,
        mn=mn,
        loss_type=loss_type,
        ot_method=ot_method,
        beta=beta,
        iter_bound=iter_bound,
        inner_iteration=inner_iteration,
        sk_bound=sk_bound,
        node_prior=node_prior,
        max_iter=max_iter,
        cost_bound=cost_bound,
        update_p=update_p,
        lr=lr,
        alpha=alpha,
        clus=clus,
        level=level,
    )
    _warn_unused(alg.name, **kwargs)
    return alg


def path(pair: _GraphPair, **kwargs: Any) -> _Path:
    """
    Path

    :param pair: Graph Pair object
    """
    alg = _Path(pair)
    _warn_unused(alg.name, **kwargs)
    return alg


def isorank(
    pair: _GraphPair,
    L: object = None,
    lalpha: float = 1.0,
    weighted: bool = True,
    alpha: float = 0.5,
    tol: float = 1e-12,
    maxiter: int = 1,
    **kwargs: Any,
) -> _Isorank:
    """
    IsoRank

    :param pair: Graph Pair object
    :param L: Pre-computed similarity matrix (default None, computed from lalpha)
    :param lalpha: Degree-based similarity scaling factor (default 1.0)
    :param weighted: Whether to use weighted similarity (default True)
    :param alpha: Teleportation probability (default 0.5)
    :param tol: Convergence tolerance (default 1e-12)
    :param maxiter: Maximum iterations (default 1)
    """
    alg = _Isorank(pair, L=L, lalpha=lalpha, weighted=weighted, alpha=alpha, tol=tol, max_iter=maxiter)
    _warn_unused(alg.name, **kwargs)
    return alg


def nsd(pair: _GraphPair, alpha: float = 0.5, iters: int = 5, **kwargs: Any) -> _NSD:
    """
    NSD (Network Similarity Decomposition)

    :param pair: Graph Pair object
    :param alpha: Damping factor (default 0.5)
    :param iters: Number of iterations (default 5)
    """
    alg = _NSD(pair, alpha=alpha, iters=iters)
    _warn_unused(alg.name, **kwargs)
    return alg


def mmnc(pair: _GraphPair, train_ratio: float = 0.04, k_de: int = 3, k_nei: int = 7, t: int = 10, fast_select: bool = False, **kwargs: Any) -> _Mmnc:
    """
    MMNC (Multi-order Matched Neighborhood Consistent Graph Alignment in a Union Vector Space)

    :param pair:
    :param train_ratio:
    :param k_de:
    :param k_nei:
    :param t:
    :param fast_select:
    :param kwargs:
    :return:
    """
    alg = _Mmnc(pair, train_ratio, k_de, k_nei, t, fast_select)
    _warn_unused(alg.name, **kwargs)
    return alg

def gradp(pair: _GraphPair, anchor_links: list | None = None, **kwargs: Any) -> _GradP:
    """
    GradP (Grad-Align+)

    :param pair: Graph Pair object
    :param anchor_links: Lost of anchor pairs in src and target graphs (optional)
    """

    if anchor_links is not None:
        anchors_src = anchor_links[:, 0].tolist() # source node indices
        anchors_tar = anchor_links[:, 1].tolist() # target node indices
    else:
        anchors_src, anchors_tar = None, None

    alg = _GradP(pair, anchors_src=anchors_src, anchors_tar=anchors_tar)
    _warn_unused(alg.name, **kwargs)
    return alg


def dspp(pair, **kwargs) -> _DSPP:
    """
    DSPP (DS++)

    :param pair: Graph Pair object
    """

    alg = _DSPP(pair=pair)
    _warn_unused(alg.name, **kwargs)
    return alg

def fgot(pair: _GraphPair, tau: int = 1, n_samples: int = 5, epochs: int = 1000, lr: int = 1, std_init: int = 5,
         loss_type: str = 'w_simple', seed: int = 42, verbose:bool = True, tol: float = 1e-12, adapt_lr: bool = False, **kwargs):
    """
    FGOT (Graph Distances Based on Filters and Optimal Transport)

    :param pair:
    :param tau:
    :param n_samples:
    :param epochs:
    :param lr:
    :param std_init:
    :param loss_type:
    :param seed:
    :param verbose:
    :param tol:
    :param adapt_lr:
    :return:
    """
    alg = _FGot(pair, tau, n_samples, epochs, lr, std_init, loss_type, seed, verbose, tol, adapt_lr)
    _warn_unused(alg.name, **kwargs)
    return alg

def got(pair: _GraphPair, it: int = 20, n_samples: int = 20, epochs: int = 10,
        lr: float = 0.1, alpha: float = 0.1, ones: bool = True, loss_type: str = 'w',
        seed: int = 42, verbose: bool = False, **kwargs):
    """
    GOT (An Optimal Transport framework for Graph comparison)

    :param pair: Graph Pair object
    :param it:
    :param n_samples:
    :param epochs:
    :param lr:
    :param alpha:
    :param ones:
    :param loss_type:
    :param seed:
    :param verbose:
    """
    alg = _Got(pair, it, n_samples, epochs, lr, alpha, ones, loss_type, seed, verbose)
    _warn_unused(alg.name, **kwargs)
    return alg


def grasp(pair: _GraphPair, n_eig: int = 100, q: int = 20, k: int = 20, laa: int = 3,
          icp: bool = True, icp_its: int = 3, lower_t: float = 0.1, upper_t: float = 50,
          linsteps: bool = True, base_align: bool = True, **kwargs) -> _Grasp:
    """
    GRASP (Scalable Graph Alignment by Spectral Corresponding Functions)

    :param pair:
    :param n_eig:
    :param q:
    :param k:
    :param laa:
    :param icp:
    :param icp_its:
    :param lower_t:
    :param upper_t:
    :param linsteps:
    :param base_align:
    :param kwargs:
    :return:
    """

    alg = _Grasp(pair, n_eig, q, k, laa, icp, icp_its, lower_t, upper_t, linsteps, base_align)
    _warn_unused(alg.name, **kwargs)
    return alg

def klaus(pair: _GraphPair, a: int = 1, b: int = 1, gamma:float = 0.4, stepm:int = 25,
          rtype: int = 1, maxiter: int = 1000, verbose: bool = False, **kwargs) -> _Klaus:
    """
    KLAUS

    :param pair:
    :param a:
    :param b:
    :param gamma:
    :param stepm:
    :param rtype:
    :param maxiter:
    :param verbose:
    :param kwargs:
    :return:
    """

    alg = _Klaus(pair, a, b, gamma, stepm, rtype, maxiter, verbose)
    _warn_unused(alg.name, **kwargs)
    return alg


def lera(pair: _GraphPair, iters: int = 30, method: str = "lowrank_svd_union",
         b_match: int = 1, default_params: bool = True, **kwargs) -> _Lrea:
    """
    LERA (Low Rank Spectral Network Alignment)

    :param pair:
    :param iters:
    :param method:
    :param b_match:
    :param default_params:
    :param kwargs:
    :return:
    """

    alg = _Lrea(pair, iters, method, b_match, default_params)
    _warn_unused(alg.name, **kwargs)
    return alg


def mds(pair: _GraphPair, n_components: int = 10, alpha: float = 0.05,
        max_iter: int = 600, eps: float = 0.01, tol:float = 1e-3,
        min_eps: float = 0.001, eps_annealing: bool = True, alpha_annealing: bool = True,
        gw_init: bool = True, return_stress: bool = False, **kwargs) -> _Mds:
    """
    MDS (Unsupervised Manifold Alignment with Joint Multidimensional Scaling)

    :param pair:
    :param n_components:
    :param alpha:
    :param max_iter:
    :param eps:
    :param tol:
    :param min_eps:
    :param eps_annealing:
    :param alpha_annealing:
    :param gw_init:
    :param return_stress:
    :param kwargs:
    :return:
    """

    alg = _Mds(pair, n_components, alpha, max_iter, eps, tol, min_eps, eps_annealing, alpha_annealing, gw_init, return_stress)
    _warn_unused(alg.name, **kwargs)
    return alg

def net_align(pair: _GraphPair, a: int = 1, b: int = 1, gamma: float = 0.5,
              dtype:int = 0, maxiter:int = 100, **kwargs) -> _NetAlign:
    """
    Net-Align

    :param pair:
    :param a:
    :param b:
    :param gamma:
    :param dtype:
    :param maxiter:
    :param kwargs:
    :return:
    """

    alg = _NetAlign(pair, a, b, gamma, dtype, maxiter)
    _warn_unused(alg.name, **kwargs)
    return alg

def parrot(pair: _GraphPair, sep_rwr_iter: int = 100, prod_rwr_iter: int = 20,
           alpha: float = 0.5, beta: float = 0.15, gamma: float = 0.5,
           in_iter: int = 50, out_iter: int = 20,
           l1: float = 0.1, l2: float = 0.1, l3: float = 0.1, l4: float = 0.01, **kwargs) -> _Parrot:
    """
    PARROT (Position-Aware Regularized Optimal Transport for Network Alignment)

    :param pair:
    :param sep_rwr_iter:
    :param prod_rwr_iter:
    :param alpha:
    :param beta:
    :param gamma:
    :param in_iter:
    :param out_iter:
    :param l1:
    :param l2:
    :param l3:
    :param l4:
    :param kwargs:
    :return:
    """

    alg = _Parrot(pair, sep_rwr_iter, prod_rwr_iter, alpha, beta, gamma, in_iter, out_iter, l1, l2, l3, l4)
    _warn_unused(alg.name, **kwargs)
    return alg


def regal(pair: _GraphPair, attributes: str | None = None, numtop: int = 10,
          alpha: float = 0.01, buckets: float = 2, k: int = 10, gammastruc: float = 1.0,
          gammaattr: float = 1.0, untillayer: int = 2, **kwargs) -> _Regal:
    """
    REGAL (Representation Learning-based Graph Alignment)

    :param pair: Graph Pair object
    :param attributes: Path to a saved numpy matrix of node attributes (None for structural only)
    :param numtop: Number of top similarities to keep via KD-tree (0 = all pairwise)
    :param alpha: Discount factor for further layers
    :param buckets: Base of log for degree binning (1 = no log binning)
    :param k: Controls the number of landmarks to sample (d = k*log(n))
    :param gammastruc: Weight on structural similarity
    :param gammaattr: Weight on attribute similarity
    :param untillayer: Calculate xNetMF until this layer (0 = no limit)
    """
    alg = _Regal(pair, attributes, numtop, alpha, buckets, k, gammastruc, gammaattr, untillayer)
    _warn_unused(alg.name, **kwargs)
    return alg


def joena(pair: _GraphPair,
    anchor_links: np.ndarray,
    hidden_dim: int = 128,
    out_dim: int = 128,
    alpha: float = 0.9,
    gamma_p: float = 1e-2,
    in_iter: int = 5,
    out_iter: int = 10,
    init_threshold_lambda: float = 1.0,
    lr: float = 1e-3,
    epochs: int = 30, runs: int = 1, device: str = 'cpu', **kwargs) -> _Joena:

    """
    JOENA (Joint Optimal Transport and Embedding for Network Alignment)

    :param pair:
    :param anchor_links:
    :param hidden_dim:
    :param out_dim:
    :param alpha:
    :param gamma_p:
    :param in_iter:
    :param out_iter:
    :param init_threshold_lambda:
    :param lr:
    :param epochs:
    :param runs:
    :param device:
    :return:
    """

    alg = _Joena(pair, anchor_links, hidden_dim, out_dim, alpha, gamma_p, in_iter, out_iter, init_threshold_lambda, lr, epochs, runs, device)
    _warn_unused(alg.name, **kwargs)
    return alg

def htc(pair: _GraphPair, src_laps: torch.Tensor | None = None,
        trg_laps: torch.Tensor | None = None, hid_dim: int = 500, num_utrn: int = 15,
        ulr: float = 0.01, num_ftune: int = 25, flr: float = 0.01, alpha: float = 1.1,
        k: int = 40, p: float = 0.5, graphlet_size: int = 4, **kwargs) -> _Htc:
    """
    HTC (Towards Higher-order Topological Consistency for Unsupervised Network Alignment)

    :param pair: Graph Pair object
    :param src_laps: Precomputed source graphlet-orbit built from pair.src when None.
    :param trg_laps: Precomputed target graphlet-orbit built from pair.tar when None.
    :param hid_dim:
    :param num_utrn:
    :param ulr:
    :param num_ftune:
    :param flr:
    :param alpha:
    :param k:
    :param p:
    :param graphlet_size:
    """

    alg = _Htc(pair, hid_dim, num_utrn, ulr, num_ftune, flr, alpha, k, p,
               src_laps=src_laps, trg_laps=trg_laps, graphlet_size=graphlet_size)
    _warn_unused(alg.name, **kwargs)
    return alg

def slot_a(pair: _GraphPair, step_size: float = 0.1, bases: int = 2,
           joint_epoch: int = 100, epoch: int = 100, gw_beta: float = 0.01, **kwargs):
    """
    SlotA

    :param pair:
    :param step_size:
    :param bases:
    :param joint_epoch:
    :param epoch:
    :param gw_beta:
    :return:
    """
    alg = _SlotA(pair, step_size, bases, joint_epoch, epoch, gw_beta)
    _warn_unused(alg.name, **kwargs)
    return alg


def alpine(pair: _GraphPair, anchor_links: Union[List[Tuple[int, int]], None] = None,k: Union[int, None] = None, connected: bool = False, mu: Union[float, None] = None, niter: int = 10, sinkhorn_iters: int = 500, sinkhorn_tol: float = 1e-9, method: Literal['FW', 'AMD'] = 'FW', **kwargs) -> _ALPINE:
    """

    :param pair:
    :param anchor_links:
    :param k:
    :param connected:
    :param mu:
    :param niter:
    :param sinkhorn_iters:
    :param sinkhorn_tol:
    :param method:
    :return:
    """
    alg = _ALPINE(pair, anchor_links, k, connected, mu, niter, sinkhorn_iters, sinkhorn_tol, method)
    _warn_unused(alg.name, **kwargs)
    return alg


def next_align(pair: _GraphPair, anchor_links: np.ndarray, dim: int = 128, num_layer: int = 1,
               coeff1: float = 1.0, coeff2: float = 1.0, lr: float = 0.01, epochs: int = 100, batch_size: int = 300,
               walks_num: int = 100, N_steps: int = 10, N_negs: int = 20, p: int = 1, q: int = 1, walk_length: int = 80,
               num_walks: int = 10, dist: str = 'L1', device: str = 'cpu', **kwargs) -> object: #_NextAlign:
    """
    Next Align, skipped due to DGL compatibility issues

    :param pair:
    :param anchor_links:
    :param dim:
    :param num_layer:
    :param coeff1:
    :param coeff2:
    :param lr:
    :param epochs:
    :param batch_size:
    :param walks_num:
    :param N_steps:
    :param N_negs:
    :param p:
    :param q:
    :param walk_length:
    :param num_walks:
    :param dist:
    :param device:
    :param kwargs:
    :return:
    """
    raise NotImplementedError

    alg = _NextAlign(pair, anchor_links, dim, num_layer, coeff1, coeff2, lr, epochs, batch_size, walks_num, N_steps, N_negs, p, q, walk_length, num_walks, dist, device)
    _warn_unused(alg.name, **kwargs)
    return alg

