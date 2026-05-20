from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from graphalign.graph import GraphPair

class Algorithm(ABC):
   """Abstract base class for all graph alignment algorithms."""

   def __init__(self, pair: GraphPair) -> None:
      self.pair = pair

   @abstractmethod
   def evaluate(self) -> np.ndarray:
      """Run the algorithm and return the alignment matrix."""

   @property
   def name(self) -> str:
      return type(self).__name__

   def __repr__(self) -> str:
      return f"{self.name}()"