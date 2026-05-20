from __future__ import annotations
from dataclasses import dataclass, field

from functools import cached_property

import networkx as nx
import numpy as np
import scipy.sparse as sps

from .._logging import logger
from ..generation.generate import apply_noise, permute_graph
from ..generation.similarities_preprocess import create_L, create_S


@dataclass
class GraphPair:
    """A matched source/target graph pair with lazily computed representations.

    Algorithms access whatever format they need (adjacency matrix, node-pair
    candidates, edge-pair adjacency) via properties, representations are
    computed once on first access and cached.

    Node features (src_features, tar_features) are optional arrays of shape
    [n, d]. When provided, algorithms can use them instead of falling back to
    structural features. The caller is responsible for ensuring feature rows
    remain consistent with node indices after any permutation or relabelling.
    """
    src: nx.Graph
    tar: nx.Graph
    ground_truth: tuple[np.ndarray, np.ndarray] | None = None
    src_features: np.ndarray | None = None  # shape [n_src, d]
    tar_features: np.ndarray | None = None  # shape [n_tar, d]


    @classmethod
    def from_graph(
        cls,
        G: nx.Graph,
        features: np.ndarray | None = None,
    ) -> GraphPair:
        """Create a graph pair from a single base graph with an identity ground truth.

        Both src and tar start as copies of G. Use .permute() and .add_noise()
        to apply transformations afterward.

        Args:
            G: Base graph to derive the pair from.
            features: Optional node feature matrix of shape [n, d], shared by
                both src and tar before any permutation is applied.
        """
        n = G.number_of_nodes()
        identity: tuple[np.ndarray, np.ndarray] = (np.arange(n), np.arange(n))
        src_feat = features.copy() if features is not None else None
        tar_feat = features.copy() if features is not None else None
        return cls(G.copy(), G.copy(), identity, src_feat, tar_feat)

    @classmethod
    def from_graphs(
        cls,
        src: nx.Graph,
        tar: nx.Graph,
        ground_truth: tuple[np.ndarray, np.ndarray] | None = None,
        src_features: np.ndarray | None = None,
        tar_features: np.ndarray | None = None,
    ) -> GraphPair:
        """Create a pair from two already-constructed graphs."""
        return cls(src, tar, ground_truth, src_features, tar_features)

    def permute(self) -> GraphPair:
        """Derive a synthetic target by randomly permuting src node labels.

        Intended use: synthetic benchmarks created via GraphPair.from_graph(G).
        Replaces self.tar with a permuted copy of self.src and sets
        self.ground_truth to the corresponding node mapping.

        Warning: Logs a warning if src and tar are not the same graph, since
        this operation discards tar. For pairs built from two distinct
        real-world graphs, use shuffle_labels() instead.

        Returns self to allow chaining.
        """
        if not nx.utils.graphs_equal(self.src, self.tar):
            logger.warning(
                "permute() replaces self.tar with a permuted copy of self.src, "
                "but src and tar appear to be different graphs. "
                "Did you mean shuffle_labels()?"
            )

        src_edges = np.array(self.src.edges())
        n = self.src.number_of_nodes()

        tar_edges, ground_truth = permute_graph(src_edges, n)

        self.tar = nx.Graph(tar_edges.tolist())
        self.tar.add_nodes_from(range(n))
        self.ground_truth = ground_truth
        if self.tar_features is not None:
            src_to_tar, tar_to_src = ground_truth
            self.tar_features = self.tar_features[tar_to_src]
        self._invalidate_cache()
        return self

    def shuffle_labels(self) -> GraphPair:
        """Randomly relabel nodes in both src and tar to remove label-based shortcuts.

        Intended use: pairs built from two distinct real-world graphs via
        GraphPair.from_graphs(src, tar). Applies independent random permutations
        to each graph's node labels so algorithms cannot exploit coincidentally
        matching graph patterns. Any existing ground truth is updated to remain valid.

        Warning: Logs a warning if src and tar are the same graph, since
        for synthetic benchmarks permute() is the appropriate operation.

        Returns self for chaining.
        """
        if nx.utils.graphs_equal(self.src, self.tar):
            logger.warning(
                "shuffle_labels() permutes both graphs independently, "
                "but src and tar appear to be the same graph. "
                "Did you mean permute()?"
            )

        n_src = self.src.number_of_nodes()
        n_tar = self.tar.number_of_nodes()
        p_src = np.random.permutation(n_src)
        p_tar = np.random.permutation(n_tar)

        self.src = nx.relabel_nodes(self.src, dict(enumerate(p_src.tolist())))
        self.tar = nx.relabel_nodes(self.tar, dict(enumerate(p_tar.tolist())))

        if self.ground_truth is not None:
            src_to_tar, tar_to_src = self.ground_truth

            new_src_to_tar = np.empty_like(src_to_tar)
            new_src_to_tar[p_src] = p_tar[src_to_tar]

            new_tar_to_src = np.empty_like(tar_to_src)
            new_tar_to_src[p_tar] = p_src[tar_to_src]

            self.ground_truth = (new_src_to_tar, new_tar_to_src)

        if self.src_features is not None:
            new_src_features = np.empty_like(self.src_features)
            new_src_features[p_src] = self.src_features
            self.src_features = new_src_features
        if self.tar_features is not None:
            new_tar_features = np.empty_like(self.tar_features)
            new_tar_features[p_tar] = self.tar_features
            self.tar_features = new_tar_features

        self._invalidate_cache()
        return self

    def add_noise(
        self,
        source_noise: float = 0.0,
        target_noise: float = 0.0,
        refill: bool = False,
    ) -> GraphPair:
        """Remove (and optionally refill) edges from src and/or tar.

        Args:
            source_noise: Fraction of source edges to remove.
            target_noise: Fraction of target edges to remove.
            refill: If True, removed edges are replaced with random ones.

        Returns self for chaining.
        """
        src_edges = np.array(self.src.edges())
        tar_edges = np.array(self.tar.edges())
        n = self.src.number_of_nodes()
        n_edges = src_edges.shape[0]

        src_edges, tar_edges = apply_noise(src_edges, tar_edges, n, n_edges, source_noise, target_noise, refill)

        self.src = nx.Graph(src_edges.tolist())
        self.tar = nx.Graph(tar_edges.tolist())
        self.src.add_nodes_from(range(n))
        self.tar.add_nodes_from(range(n))
        self._invalidate_cache()
        return self

    def _invalidate_cache(self) -> None:
        """Discard all cached representations so they are recomputed on next access."""
        for attr in ("src_adjacency", "tar_adjacency", "L", "S", "_L_coo"):
            self.__dict__.pop(attr, None)

    # --- Core representations ---

    @cached_property
    def src_adjacency(self) -> np.ndarray:
        """n x n adjacency matrix for src, rows/cols ordered 0..n-1."""
        n = self.src.number_of_nodes()
        return nx.to_numpy_array(self.src, nodelist=range(n))

    @cached_property
    def tar_adjacency(self) -> np.ndarray:
        """n x n adjacency matrix for tar, rows/cols ordered 0..n-1."""
        n = self.tar.number_of_nodes()
        return nx.to_numpy_array(self.tar, nodelist=range(n))

    # --- Preprocessing

    @cached_property
    def L(self) -> sps.csr_matrix:
        """Candidate match scores: L[i, j] = how likely src node i aligns to tar node j (degree-based)."""
        return create_L(self.src_adjacency, self.tar_adjacency)

    @cached_property
    def S(self) -> sps.csr_matrix:
        """Edge-pair adjacency over L's non-zeros: S[e1, e2] = 1 if edges e1 and e2 share a matched endpoint."""
        return create_S(
            sps.csr_matrix(self.src_adjacency),
            sps.csr_matrix(self.tar_adjacency),
            self.L,
        )

    @cached_property
    def _L_coo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        li, lj, w = sps.find(self.L)
        return li, lj, w

    @property
    def li(self) -> np.ndarray:
        """Src node indices of L's non-zero entries."""
        return self._L_coo[0]

    @property
    def lj(self) -> np.ndarray:
        """Tar node indices of L's non-zero entries."""
        return self._L_coo[1]

    @property
    def w(self) -> np.ndarray:
        """Weights of L's non-zero entries."""
        return self._L_coo[2]

    def get_anchor_links(self, percentage) -> np.ndarray:
        """
        Selects a percentage of random nodes to return as an anchors

        :param percentage: Percentage of hole to be takes as anchors
        :return: anchors as a np.ndarray
        """

        src_to_tar, _ = self.ground_truth

        # Build the full correspondence as (src_node, tar_node) rows
        all_pairs = np.column_stack([
            np.arange(len(src_to_tar)),  # src node indices
            src_to_tar  # corresponding tar node indices
        ])

        rng = np.random.default_rng(42)
        k = max(1, int(percentage * len(all_pairs)))
        anchor_links = all_pairs[rng.choice(len(all_pairs), size=k, replace=False)]
        return anchor_links