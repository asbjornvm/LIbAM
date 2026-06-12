import networkx as nx
import numpy as np
import pytest
import scipy.optimize

import libam.algorithms as alg
from libam.graph.graph_pair import GraphPair
from libam.evaluation import accuracy


N_SEEDS = 5


def _make_pair(seed: int) -> GraphPair:
    np.random.seed(seed)
    graph = nx.barabasi_albert_graph(64, 4)
    return GraphPair.from_graph(graph).permute().add_noise(0.05)


def _acc(p: np.ndarray, pair: GraphPair) -> float:
    acc = accuracy(pair, p)
    return acc


def _passes_once(fn: callable, n_seeds: int = N_SEEDS) -> bool:
    for seed in range(n_seeds):
        pair = _make_pair(seed)
        if _acc(fn(pair).align(), pair) != 0.0:
            return True
    return False


def test_fugal():     assert _passes_once(alg.fugal)
def test_grampa():    assert _passes_once(alg.grampa)
def test_grampa_s():  assert _passes_once(alg.grampa_s)
def test_cone():      assert _passes_once(alg.cone)
def test_gwl():       assert _passes_once(alg.gwl)
def test_path():      assert _passes_once(alg.path)
def test_isorank():   assert _passes_once(alg.isorank)
def test_nsd():       assert _passes_once(alg.nsd)
def test_gradp():     assert _passes_once(alg.gradp)
def test_dspp():      assert _passes_once(alg.dspp)
def test_grasp():     assert _passes_once(alg.grasp)
def test_klaus():     assert _passes_once(alg.klaus)
def test_lera():      assert _passes_once(alg.lera)
def test_mds():       assert _passes_once(alg.mds)
def test_net_align(): assert _passes_once(alg.net_align)
def test_parrot():    assert _passes_once(alg.parrot)

def test_joena():
    def joena_with_anchors(pair: GraphPair):
        return alg.joena(pair, anchor_links=pair.get_anchor_links(0.1))
    assert _passes_once(joena_with_anchors)

def test_htc():       assert _passes_once(alg.htc)
