import logging
from typing import Any

import numpy as np
import torch

from graphalign.algorithms.unrestricted.FUGAL.Fugal import Fugal as _Fugal
from graphalign.algorithms.unrestricted.GrampaS.GrampaS import GrampaS as _GrampaS
from graphalign.algorithms.unrestricted.Grampa.Grampa import Grampa as _Grampa
from graphalign.algorithms.unrestricted.GWL.gwl import GWL as _GWL
from graphalign.algorithms.unrestricted.GWL.sgwl import SGWL as _SGWL
from graphalign.algorithms.unrestricted.CONE.conealign import Cone as _Cone
from graphalign.algorithms.unrestricted.NSD.NSD import NSD as _NSD
from graphalign.algorithms.unrestricted.Path.path import Path as _Path
from graphalign.algorithms.unrestricted.mcmc.mc import Mcmc as _Mcmc
from graphalign.algorithms.unrestricted.DSPP.Dspp import DSPP as _DSPP
from graphalign.algorithms.unrestricted.GrASp.grasp import Grasp as _Grasp
from graphalign.algorithms.unrestricted.KLAUS.klaus import Klaus as _Klaus
from graphalign.algorithms.unrestricted.LREA.lera import Lera as _Lera
from graphalign.algorithms.unrestricted.MDS.mds import Mds as _Mds
from graphalign.algorithms.unrestricted.NetAlign.netalign import NetAlign as _NetAlign
from graphalign.algorithms.unrestricted.REGAL.regal import Regal as _Regal
from graphalign.algorithms.unrestricted.Parrot.Parrot import Parrot as _Parrot
from graphalign.algorithms.restricted.GradP.gradp import GradP as _GradP
from graphalign.algorithms.unrestricted.isorank.isorank import Isorank as _Isorank
from graphalign.algorithms.restricted.JOENA.joena import Joena as _Joena
from graphalign.algorithms.restricted.HTC.htc import HTC as _Htc
from graphalign.algorithms.restricted.SlotA.slot_a import SlotA as _SlotA

from graphalign.graph import GraphPair
from graphalign.algorithms.algorithm import Algorithm

logger = logging.getLogger(__name__)

def _warn_unused(name:str, **kwargs) -> None:
    for key in kwargs:
        logger.warning(f"{name} received unknown kwarg '{key}' - it will be ignored")


def fugal(pair: GraphPair, iterations: int = 1, simple: bool = True, mu: float = 0.05, efn: int = 3, **kwargs: Any) -> _Fugal:
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


def grampa_s(pair: GraphPair, eta: float = 0.2, init_sim: int = 1, eig_type: int = 0, **kwargs: Any) -> _GrampaS:
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


def grampa(pair: GraphPair, eta: float = 0.2, **kwargs: Any) -> _Grampa:
    """
    GRAMPA

    :param pair: Graph Pair object
    :param eta: Regularisation parameter
    """
    alg = _Grampa(pair, eta=eta)
    _warn_unused(alg.name, **kwargs)
    return alg


def cone(
    pair: GraphPair,
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
    CONE

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
    pair: GraphPair,
    # Optimiser options
    epochs: int = 1,
    batch_size: int = 64,
    use_cuda: bool = True,
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
    pair: GraphPair,
    ot_dict: dict,
    mn: int,
    max_cpu: int = 0,
    clus: int = 2,
    level: int = 3,
    **kwargs: Any,
) -> _SGWL:
    """
    SGWL (Scalable Gromov-Wasserstein Learning)

    :param pair: Graph Pair object
    :param ot_dict: Optimal transport config
    :param mn: Min cluster size
    :param max_cpu: Max CPU threads (default 0 = all)
    :param clus: Number of clusters (default 2)
    :param level: Hierarchy levels (default 3)
    """
    alg = _SGWL(pair, ot_dict=ot_dict, mn=mn, max_cpu=max_cpu, clus=clus, level=level)
    _warn_unused(alg.name, **kwargs)
    return alg


def path(pair: GraphPair, **kwargs: Any) -> _Path:
    """
    Path

    :param pair: Graph Pair object
    """
    alg = _Path(pair)
    _warn_unused(alg.name, **kwargs)
    return alg


def isorank(
    pair: GraphPair,
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


def nsd(pair: GraphPair, alpha: float = 0.5, iters: int = 5, **kwargs: Any) -> _NSD:
    """
    NSD (Network Similarity Decomposition)

    :param pair: Graph Pair object
    :param alpha: Damping factor (default 0.5)
    :param iters: Number of iterations (default 5)
    """
    alg = _NSD(pair, alpha=alpha, iters=iters)
    _warn_unused(alg.name, **kwargs)
    return alg


def mcmc(pair: GraphPair, train_ratio: float = 0.04, k_de: int = 3, k_nei: int = 7, t: int = 10, fast_select: bool = False, **kwargs: Any) -> _Mcmc:
    """
    MCMC

    Current cannot run with 128 nodes as there is a < that should be an <=

    :param pair:
    :param train_ratio:
    :param k_de:
    :param k_nei:
    :param t:
    :param fast_select:
    :param kwargs:
    :return:
    """
    alg = _Mcmc(pair, train_ratio, k_de, k_nei, t, fast_select)
    _warn_unused(alg.name, **kwargs)
    return alg

def gradp(pair: GraphPair, anchor_links: list | None = None, **kwargs: Any) -> _GradP:
    """
    GradP

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
    DSPP

    :param pair: Graph Pair object
    """

    alg = _DSPP(pair=pair)
    _warn_unused(alg.name, **kwargs)
    return alg

def fgot(pair: GraphPair, tau: int = 1, n_samples: int = 5, epochs: int = 1000, lr: int = 1, std_init: int = 5,
         loss_type: str = 'w_simple', seed: int = 42, verbose:bool = True, tol: float = 1e-12, adapt_lr: bool = False, **kwargs):
    """
    FGOT
    TODO: fill parm descriptions

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
    raise NotImplementedError # need to get the def on a function to add this one to the lib
    alg = _fgot(pair, tau, n_samples, epochs, lr, std_init, loss_type, seed, verbose, tol, adapt_lr)
    _warn_unused(alg.name, **kwargs)
    return alg

def grasp(pair: GraphPair, n_eig: int = 100, q: int = 20, k: int = 20, laa: int = 3,
          icp: bool = True, icp_its: int = 3, lower_t: float = 0.1, upper_t: float = 50,
          linsteps: bool = True, base_align: bool = True, **kwargs) -> _Grasp:
    """
    GRASP
    # TODO fill docstring

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

def klaus(pair: GraphPair, a: int = 1, b: int = 1, gamma:float = 0.4, stepm:int = 25,
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


def lera(pair: GraphPair, iters: int = 30, method: str = "lowrank_svd_union",
         b_match: int = 1, default_params: bool = True, **kwargs) -> _Lera:
    """
    LERA

    :param pair:
    :param iters:
    :param method:
    :param b_match:
    :param default_params:
    :param kwargs:
    :return:
    """

    alg = _Lera(pair, iters, method, b_match, default_params)
    _warn_unused(alg.name, **kwargs)
    return alg


def mds(pair: GraphPair, n_components: int = 2, alpha: float = 1.0,
        max_iter: int = 300, eps: float = 0.01, tol:float = 1e-3,
        min_eps: float = 0.001, eps_annealing: bool = True, alpha_annealing: bool = False,
        gw_init: bool = False, return_stress: bool = False, **kwargs) -> _Mds:
    """
    MDS

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

def net_align(pair: GraphPair, a: int = 1, b: int = 1, gamma: float = 0.5,
              dtype:int = 0, maxiter:int = 100, verbose: bool = False, **kwargs) -> _NetAlign:

    alg = _NetAlign(pair, a, b, gamma, dtype, maxiter, verbose)
    _warn_unused(alg.name, **kwargs)
    return alg

def parrot(pair: GraphPair, sep_rwr_iter: int = 100, prod_rwr_iter: int = 20,
           alpha: float = 100, beta: float = 50, gamma: float = 0.5,
           in_iter: int = 0.15, out_iter: int = 0.5,
           l1: float = 0.1, l2: float = 0.1, l3: float = 0.1, l4: float = 0.01, **kwargs) -> _Parrot:
    #TODO: add param info

    alg = _Parrot(pair, sep_rwr_iter, prod_rwr_iter, alpha, beta, gamma, in_iter, out_iter, l1, l2, l3, l4)
    _warn_unused(alg.name, **kwargs)
    return alg


def regal(pair: GraphPair, **kwargs) -> _Regal:
    raise NotImplementedError

    alg = _Regal(pair)
    _warn_unused(alg.name, **kwargs)
    return alg


def joena(pair: GraphPair,
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
    JOENA

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

def htc(pair: GraphPair, src_laps: torch.Tensor, trg_laps: torch.Tensor,
        hid_dim: int = 500, num_utrn: int = 15, ulr: float = 0.01, num_ftune: int = 25,
        flr: float = 0.01, alpha: float = 1.1, k: int = 40, p: float = 0.5, **kwargs) -> _Htc:

    """
    HTC

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

    raise NotImplementedError

    alg = _Htc(pair, src_laps, trg_laps, hid_dim, num_utrn, ulr, num_ftune, flr, alpha, k, p)
    _warn_unused(alg.name, **kwargs)
    return alg

def slot_a(pair: GraphPair, step_size: float = 0.1, bases: int = 2,
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

    raise NotImplementedError

    alg = _SlotA(pair, step_size, bases, joint_epoch, epoch, gw_beta)
    _warn_unused(alg.name, **kwargs)
    return alg


def alpine(pair: GraphPair) -> Algorithm:
    raise NotImplementedError