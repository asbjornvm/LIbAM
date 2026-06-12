from dataclasses import dataclass

import numpy as np
import scipy
import torch

from libam.algorithms.algorithm import AlignAlgorithm
from libam.graph.graph_pair import GraphPair

from . import xnetmf
from .alignments import get_embedding_similarities, get_embeddings
from .config import Graph, RepMethod

# original code from https://github.com/GemsLab/REGAL


def _g_to_adj(src: np.ndarray, tar: np.ndarray) -> np.ndarray:
    """Embed two adjacency matrices into a single block-diagonal adjacency."""
    adj1 = np.kron(np.array([[1, 0], [0, 0]], dtype=np.int8), src)
    adj2 = np.kron(np.array([[0, 0], [0, 1]], dtype=np.int8), tar)
    return (adj1 + adj2).clip(0, 1)


@dataclass
class Regal(AlignAlgorithm):
    pair: GraphPair
    attributes: str | None = None
    numtop: int = 10
    alpha: float = 0.01
    buckets: float = 2
    k: int = 10
    gammastruc: float = 1.0
    gammaattr: float = 1.0
    untillayer: int = 2

    @property
    def name(self) -> str:
        return "REGAL"

    def _align(self) -> np.ndarray | torch.Tensor | scipy.sparse.csr_matrix:
        adj = _g_to_adj(self.pair.src_adjacency, self.pair.tar_adjacency)

        representations = self._learn_representations(adj)
        emb1, emb2 = get_embeddings(representations)

        num_top = self.numtop if self.numtop != 0 else None
        alignment_matrix, _ = get_embedding_similarities(emb1, emb2, num_top=num_top)
        return alignment_matrix.toarray()

    def _learn_representations(self, adj: np.ndarray) -> np.ndarray:
        node_attributes = (
            np.load(self.attributes) if self.attributes is not None else None
        )
        graph = Graph(adj, node_attributes=node_attributes)

        max_layer = self.untillayer if self.untillayer != 0 else None
        num_buckets = self.buckets if self.buckets != 1 else None

        rep_method = RepMethod(
            max_layer=max_layer,
            alpha=self.alpha,
            k=self.k,
            num_buckets=num_buckets,
            normalize=True,
            gammastruc=self.gammastruc,
            gammaattr=self.gammaattr,
        )
        return xnetmf.get_representations(graph, rep_method)
