"""Function for synchronizing features and graphs
in the presence of anchor pairs."""
import networkx as nx
import numpy as np
from typing import List, Tuple


def synchronize_features(attr1: np.ndarray,
                         attr2: np.ndarray,
                         anchor_pairs: np.ndarray):
    """
    Synchronize features based on anchors.

    Parameters
    ----------
    attr1 : np.ndarray
        Node-attributes of G1.
    attr2 : np.ndarray
        Node-attributes of G2.
    anchor_pairs : np.ndarray
        Anchor-pairs.

    Returns
    -------
    attr1_new, attr1_new : np.ndarray
        Synchronized node-attributes for G1, G2
    """
    attr1_new = attr1.copy()
    attr2_new = attr2.copy()
    for i, j in anchor_pairs:
        # attr2_new[j] = attr1_new[i]
        attr1_new[i] = attr2_new[j]
    return attr1_new, attr2_new


def seed_link(G1: nx.Graph, G2: nx.Graph,
              anchor_pairs: np.ndarray):
    """
    Create the same links among the seed nodes in G1 and G2.

    Parameters
    ----------
    G1: nx.Graph
        Source graph G1.
    G2: nx.Graph
        Target graph G2.
    anchor_pairs: np.ndarray
        Anchor-pairs.
    """
    for k in range(len(anchor_pairs) - 1):
        i_k, j_k = anchor_pairs[k]
        for t in range(k+1, len(anchor_pairs)):
            i_t, j_t = anchor_pairs[t]
            edge_between_i = G1.has_edge(i_k, i_t)
            edge_between_j = G2.has_edge(j_k, j_t)
            if edge_between_i and not edge_between_j:
                G2.add_edge(j_k, j_t)
            elif edge_between_j and not edge_between_i:
                G1.add_edge(i_k, i_t)
