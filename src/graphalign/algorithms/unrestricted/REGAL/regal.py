import numpy as np
import os
from dataclasses import dataclass

from graphalign.algorithms.algorithm import Algorithm
from graphalign import GraphPair

#original code from https://github.com/GemsLab/REGAL
try:
    import cPickle as pickle
except ImportError:
    import pickle

from . import xnetmf
from .config import RepMethod, Graph
from .alignments import get_embeddings, get_embedding_similarities

def G_to_Adj(G1, G2):
    # adj1 = sps.kron([[1, 0], [0, 0]], G1)
    # adj2 = sps.kron([[0, 0], [0, 1]], G2)
    adj1 = np.array([[1, 0], [0, 0]], dtype=np.int8)
    adj1 = np.kron(adj1, G1)
    adj2 = np.array([[0, 0], [0, 1]], dtype=np.int8)
    adj2 = np.kron(adj2, G2)
    adj = adj1 + adj2
    # adj.data = adj.data.clip(0, 1)
    adj = adj.clip(0, 1)
    return adj

@dataclass
class Regal(Algorithm):
    pair: GraphPair
    attributes: str | None
    numtop: int
    alpha: float
    buckets: float
    k: int
    gammastruc: float
    gammaattr: float

    @property
    def name(self) -> str:
        return "REGAL"

    def evaluate(self) -> np.ndarray:
        Src = self.pair.src_adjacency
        Tar = self.pair.tar_adjacency

        # TODO: Move thread controll
        os.environ["MKL_NUM_THREADS"] = "20"
        os.environ["OMP_NUM_THREADS"] = "20"
        adj = G_to_Adj(Src, Tar)
        if self.attributes is not None:
            # load vector of attributes in from file
            self.attributes = np.load(self.attributes)

        embed = learn_representations(adj, self.alpha, self.buckets, self.k, self.gammastruc, self.gammaattr)
        emb1, emb2 = get_embeddings(embed)
        if self.numtop == 0:
            self.numtop = None
        alignment_matrix, cost_matrix = get_embedding_similarities(
            emb1, emb2, num_top=self.numtop)
        return alignment_matrix


# Should take in a file with the input graph as edgelist (REGAL_args['input)
# Should save representations to REGAL_args['output
def learn_representations(adj, attributes, untillayer, alpha, buckets, k, gammastruc, gammaattr):
    graph = Graph(adj, node_attributes=attributes)
    max_layer = untillayer
    if untillayer == 0:
        max_layer = None
    alpha = alpha
    num_buckets = buckets  # BASE OF LOG FOR LOG SCALE
    if num_buckets == 1:
        num_buckets = None
    rep_method = RepMethod(max_layer=max_layer,
                           alpha=alpha,
                           k=k,
                           num_buckets=num_buckets,
                           normalize=True,
                           gammastruc=gammastruc,
                           gammaattr=gammaattr)
    if max_layer is None:
        max_layer = 1000
    representations = xnetmf.get_representations(graph, rep_method)
    return representations


# pickle.dump(representations, open(REGAL_args['output, "w"))


def recovery(gt1, mb):
    nodes = len(gt1)
    count = 0
    for i in range(nodes):
        if gt1[i] == mb[i]:
            count = count + 1
    return count / nodes
