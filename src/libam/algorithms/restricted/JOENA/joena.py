from dataclasses import dataclass

import numpy as np
import scipy
import torch

from libam._logging import logger
from libam.graph.graph_pair import GraphPair
from libam.algorithms.algorithm import AlignAlgorithm
from .utils.dataset import build_nx_graph, build_tg_graph
from .utils.distance import rwr_scores
from .model import MLP, FusedGWLoss


@dataclass
class Joena(AlignAlgorithm):
    pair: GraphPair
    anchor_links: np.ndarray
    # Model
    hidden_dim: int
    out_dim: int
    # Loss
    alpha: float
    gamma_p: float
    in_iter: int
    out_iter: int
    init_threshold_lambda: float
    # Training
    lr: float
    epochs: int
    runs: int
    # Device
    device: str

    def __post_init__(self) -> None:
        self.__name__: str = "JOENA"

    def _align(self) -> np.ndarray | torch.Tensor | scipy.sparse.csr_matrix:
        assert torch.cuda.is_available() or self.device == 'cpu', 'CUDA is not available'
        torch_device = torch.device(self.device)
        torch.set_default_dtype(torch.float64)

        edge_index1 = np.array(self.pair.src.edges(), dtype=np.int64)
        edge_index2 = np.array(self.pair.tar.edges(), dtype=np.int64)
        x1 = self.pair.src_features
        x2 = self.pair.tar_features

        anchor1, anchor2 = self.anchor_links[:, 0], self.anchor_links[:, 1]
        G1, G2 = build_nx_graph(edge_index1, anchor1, x1), build_nx_graph(edge_index2, anchor2, x2)
        rwr1, rwr2 = rwr_scores(G1, G2, self.anchor_links, dtype=np.float64)

        x1 = rwr1 if x1 is None else np.concatenate([x1, rwr1], axis=1)
        x2 = rwr2 if x2 is None else np.concatenate([x2, rwr2], axis=1)

        G1_tg = build_tg_graph(edge_index1, x1, rwr1, dtype=torch.float64).to(torch_device)
        G2_tg = build_tg_graph(edge_index2, x2, rwr2, dtype=torch.float64).to(torch_device)
        n1, n2 = G1_tg.x.shape[0], G2_tg.x.shape[0]
        gw_weight = self.alpha / (1 - self.alpha) * min(n1, n2) ** 0.5

        for run in range(self.runs):
            logger.debug(f"Run {run + 1}/{self.runs}")
            model = MLP(input_dim=G1_tg.x.shape[1],
                        hidden_dim=self.hidden_dim,
                        output_dim=self.out_dim).to(torch_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            criterion = FusedGWLoss(G1_tg, G2_tg, anchor1, anchor2,
                                    gw_weight=gw_weight,
                                    gamma_p=self.gamma_p,
                                    init_threshold_lambda=self.init_threshold_lambda,
                                    in_iter=self.in_iter,
                                    out_iter=self.out_iter,
                                    total_epochs=self.epochs).to(torch_device)

            logger.info("Training...")
            for epoch in range(self.epochs):
                model.train()
                optimizer.zero_grad()
                out1, out2 = model(G1_tg, G2_tg)
                loss, similarity, threshold_lambda = criterion(out1=out1, out2=out2)
                loss.backward()
                optimizer.step()
                logger.debug(f'Epoch {epoch + 1}, Loss: {loss.item():.6f}')

            return similarity