"""Structural features of Alpine."""
import numpy as np
import networkx as nx


def structural_features(G):
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    degs = [node_degree_dict[n] for n in node_list]

    node_features = np.zeros(shape=(G.number_of_nodes(), 2))
    node_features[:, 0] = degs
    node_features = np.nan_to_num(node_features)
    egonets = {n: nx.ego_graph(G, n) for n in node_list}
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]
    node_features[:, 1] = neighbor_degs
    return np.nan_to_num(node_features)
