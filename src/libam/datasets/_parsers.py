import networkx as nx
import numpy as np

import libam


def parse_edge_list(graph: nx.Graph) -> libam.GraphPair:
    return libam.GraphPair.from_graph(graph)


def parse_alpine(data: tuple[nx.Graph, nx.Graph, tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]) -> libam.GraphPair:
      src, tar, ground_truth, src_features, tar_features = data
      return libam.GraphPair.from_graphs(src, tar, ground_truth, src_features, tar_features)
