import networkx as nx
import numpy as np
import pytest
import scipy.optimize

import graphalign.algorithms as alg
from graphalign import GraphPair, eval_align


@pytest.fixture(scope="module")
def pair() -> GraphPair:
    graph = nx.barabasi_albert_graph(128, 4)
    return GraphPair.from_graph(graph).permute().add_noise(0.05)


def _acc(P: np.ndarray, pair: GraphPair) -> float:
    ma, mb = scipy.optimize.linear_sum_assignment(-P)
    _, acc, _ = eval_align(ma, mb, pair.ground_truth[0])
    return acc


def test_fugal(pair):     assert _acc(alg.fugal(pair).evaluate(), pair) != 0.00
def test_grampa(pair):    assert _acc(alg.grampa(pair).evaluate(), pair) != 0.00
def test_grampa_s(pair):  assert _acc(alg.grampa_s(pair).evaluate(), pair) != 0.00
def test_cone(pair):      assert _acc(alg.cone(pair).evaluate(), pair) != 0.00
def test_gwl(pair):
    opt = {'outer_iteration': 3, 'inner_iteration': 3, 'sgd_iteration': 10, 'epochs': 1, 'batch_size': 64, 'use_cuda': False, 'strategy': 'hard', 'beta': 0.1, 'display': False, 'prefix': '', 'prior': False}
    assert _acc(alg.gwl(pair, opt_dict=opt).evaluate(), pair) != 0.00
def test_path(pair):      assert _acc(alg.path(pair).evaluate(), pair) != 0.00
def test_isorank(pair):   assert _acc(alg.isorank(pair).evaluate(), pair) != 0.00
def test_nsd(pair):       assert _acc(alg.nsd(pair).evaluate(), pair) != 0.00
def test_gradp(pair):     assert _acc(alg.gradp(pair).evaluate(), pair) != 0.00
def test_dspp(pair):      assert _acc(alg.dspp(pair).evaluate(), pair) != 0.00
def test_grasp(pair):     assert _acc(alg.grasp(pair).evaluate(), pair) != 0.00
def test_klaus(pair):     assert _acc(alg.klaus(pair).evaluate(), pair) != 0.00
def test_lera(pair):      assert _acc(alg.lera(pair).evaluate(), pair) != 0.00
def test_mds(pair):       assert _acc(alg.mds(pair).evaluate(), pair) != 0.00
def test_net_align(pair): assert _acc(alg.net_align(pair).evaluate(), pair) != 0.00
def test_parrot(pair):    assert _acc(alg.parrot(pair).evaluate(), pair) != 0.00

def test_joena(pair):
    anchors = pair.get_anchor_links(0.1)
    assert _acc(alg.joena(pair, anchor_links=anchors).evaluate(), pair) > 0.8
