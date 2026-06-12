from pathlib import Path
import networkx as nx
import numpy as np


def _load_ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.loadtxt(path, dtype=int)
    src_to_tar = np.empty(pairs[:, 0].max() + 1, dtype=int)
    tar_to_src = np.empty(pairs[:, 1].max() + 1, dtype=int)
    src_to_tar[pairs[:, 0]] = pairs[:, 1]
    tar_to_src[pairs[:, 1]] = pairs[:, 0]
    return src_to_tar, tar_to_src

def load_edge_list(path: Path) -> nx.Graph:
    g = nx.read_edgelist(str(path), nodetype=int)
    g.remove_edges_from(nx.selfloop_edges(g))
    return nx.convert_node_labels_to_integers(g, first_label=0, ordering="sorted")


def load_alpine(path: Path, name: str, features: bool = True) -> tuple[nx.Graph, nx.Graph, tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """Load a graph pair from the Alpine dataset format.

    Expects the following files in the provided folder path:
      - {name}_s_edge.txt  source edge list
      - {name}_t_edge.txt  target edge list
      - {name}_s_feat.txt  source node features (one row per node)
      - {name}_t_feat.txt  target node features (one row per node)
      - {name}_ground_True.txt  ground truth as src_node tar_node pairs
    """
    src = load_edge_list(path / f"{name}_s_edge.txt")
    tar = load_edge_list(path / f"{name}_t_edge.txt")
    ground_truth = _load_ground_truth(path / f"{name}_ground_True.txt")

    if features:
        src_features = np.loadtxt(path / f"{name}_s_feat.txt")
        tar_features = np.loadtxt(path / f"{name}_t_feat.txt")

        # read_edgelist drops isolated nodes and re-add them so node count matches feature rows
        src.add_nodes_from(range(len(src_features)))
        tar.add_nodes_from(range(len(tar_features)))
    else:
        src_features = tar_features = None

    return src, tar, ground_truth, src_features, tar_features
