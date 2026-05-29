from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
import torch
import scipy.sparse as sps

from graphalign.algorithms.utils import doubly_stochastic

if TYPE_CHECKING:
    from graphalign.graph import GraphPair


logger = logging.getLogger(__name__)


class Algorithm(ABC):
   """Abstract base class for all graph alignment algorithms."""

   def __init__(self, pair: GraphPair) -> None:
      self.execution_time: float = float('nan')
      self.pair: GraphPair = pair

   def evaluate(self) -> np.ndarray:
      start = time.perf_counter()
      result = self._evaluate()
      self.execution_time: float = time.perf_counter() - start
      logger.info(f"Converting output matrix of {self.name} to a doubly stochastic matrix")
      return doubly_stochastic(result, 20)

   @abstractmethod
   def _evaluate(self) -> np.ndarray | torch.Tensor | scipy.sparse.csr_matrix:
      """Run the algorithm and return the raw alignment matrix."""

   @property
   def name(self) -> str:
      return type(self).__name__

   def __repr__(self) -> str:
      return f"{self.name}()"