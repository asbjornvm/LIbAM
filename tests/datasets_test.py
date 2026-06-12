import os

import networkx as nx
import numpy as np

import libam.datasets as datasets
from libam.datasets._dataset import Dataset
from libam.datasets._registry import _REGISTRY, fetch
from libam.graph.graph_pair import GraphPair


def _datasets() -> list[Dataset]:
    return [obj for obj in vars(datasets).values() if isinstance(obj, Dataset)]


def _features_valid(features: np.ndarray | None, graph: nx.Graph) -> bool:
    if features is None:
        return True
    return features.ndim == 2 and features.shape[0] == graph.number_of_nodes()


def _ground_truth_valid(ground_truth: tuple[np.ndarray, np.ndarray]) -> bool:
    src_to_tar, tar_to_src = ground_truth
    return (src_to_tar.ndim == 1 and tar_to_src.ndim == 1
            and src_to_tar.size > 0 and tar_to_src.size > 0
            and np.issubdtype(src_to_tar.dtype, np.integer)
            and np.issubdtype(tar_to_src.dtype, np.integer))


def test_registry_files_download():
    """Every registered file downloads from GitHub with a matching hash."""
    for filename in _REGISTRY.registry:
        path = fetch(filename)  # pooch verifies the sha256 on download
        assert os.path.getsize(path) > 0


def test_datasets_parse_valid():
    """Every exported dataset retrieves and parses into a valid GraphPair."""
    for dataset in _datasets():
        pair = dataset.graphpair()
        assert isinstance(pair, GraphPair)
        assert pair.src.number_of_nodes() > 0 and pair.src.number_of_edges() > 0
        assert pair.tar.number_of_nodes() > 0 and pair.tar.number_of_edges() > 0
        assert pair.ground_truth is not None and _ground_truth_valid(pair.ground_truth)
        assert _features_valid(pair.src_features, pair.src)
        assert _features_valid(pair.tar_features, pair.tar)
