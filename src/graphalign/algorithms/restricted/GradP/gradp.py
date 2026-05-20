from dataclasses import dataclass

import numpy as np
import random
import torch

from graphalign.graph import GraphPair
from graphalign.algorithms.algorithm import Algorithm
from .utils import augment_attributes, aug_trimming
from .models import GradAlign


def _set_seeds(n: int) -> None:
    seed = int(n)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_perm_matrix(src_nodes: np.ndarray, tar_nodes: np.ndarray, n: int, m: int) -> np.ndarray:
    """Build a permutation matrix from paired src/tar node index arrays.

    P[src_node, tar_node] = 1 for each matched pair.
    """
    size = max(n, m)
    P = np.zeros((size, size))
    for src, tar in zip(src_nodes, tar_nodes):
        if src < size and tar < size:
            P[src, tar] = 1.0
    return P


@dataclass
class GradP(Algorithm):
    pair: GraphPair
    anchors_src: list | None = None
    anchors_tar: list | None = None

    def evaluate(self) -> np.ndarray:
        G1 = self.pair.src.copy()
        G2 = self.pair.tar.copy()
        n = G1.number_of_nodes()
        m = G2.number_of_nodes()

        for node in G1.nodes():
            if G1.degree(node) == 0:
                G1.add_edge(node, node)
        for node in G2.nodes():
            if G2.degree(node) == 0:
                G2.add_edge(node, node)

        attr1 = self.pair.src_features if self.pair.src_features is not None else np.ones((n, 1))
        attr2 = self.pair.tar_features if self.pair.tar_features is not None else np.ones((m, 1))

        idx1_dict = {i: i for i in range(n)}
        idx2_dict = {i: i for i in range(m)}
        alignment_dict = idx2_dict
        alignment_dict_reversed = idx2_dict

        attr1_aug, attr2_aug = augment_attributes(
            G1, G2, attr1, attr2,
            num_attr=15,
            version='katz',
            khop=1,
            penalty=0.1,
            normalize=True,
        )
        attr1_aug, attr2_aug = aug_trimming(attr1_aug, attr2_aug)

        train_ratio = 0.15 if self.anchors_src is not None else 0.0

        aligner = GradAlign(
            G1, G2, attr1, attr2, attr1_aug, attr2_aug,
            k_hop=3,
            hid=100,
            alignment_dict=alignment_dict,
            alignment_dict_reversed=alignment_dict_reversed,
            train_ratio=train_ratio,
            idx1_dict=idx1_dict,
            idx2_dict=idx2_dict,
            anchors_GQ=self.anchors_tar,
            anchors_G=self.anchors_src,
            alpha=m / n,
            beta=1,
        )
        seed_list1, seed_list2 = aligner.run_algorithm()

        seed_list1 = np.array(seed_list1)
        seed_list2 = np.array(seed_list2)

        # Remove duplicate target assignments, keeping first occurrence
        seen: set = set()
        mask = []
        for x in seed_list2:
            if x in seen:
                mask.append(False)
            else:
                seen.add(x)
                mask.append(True)
        mask_arr = np.array(mask)
        seed_list1 = seed_list1[mask_arr]
        seed_list2 = seed_list2[mask_arr]

        sorted_indices = np.argsort(seed_list2)
        tar_nodes = seed_list2[sorted_indices]
        src_nodes = seed_list1[sorted_indices]


        return _build_perm_matrix(src_nodes, tar_nodes, n, m)
