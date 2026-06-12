from pathlib import Path
from typing import Any, Callable

import networkx as nx

import libam
from libam.datasets._registry import fetch


class Dataset:
    def __init__(self,
                 filename: str,
                 loader: Callable[[Path], Any],
                 parser: Callable[..., libam.GraphPair],
                 members: list[str] | None = None,
                 ) -> None:
        """
        Loads a dataset file into a networkx graph object.

        :param filename: Registry key of the file to load, or the folder prefix
            (ending in ``/``) for multi-file datasets.
        :param loader: Loader function turning the resolved path into graph data.
        :param parser: Parser function turning the loaded data into a GraphPair.
        :param members: For folder datasets, the file names inside ``filename`` that
            must be fetched; the loader receives the folder path.
        :returns: simple graph object, or tuple of graph objects and their ground truth, or graph objects, features, and ground truth
        """
        self._filename = filename
        self._loader = loader
        self._parser = parser
        self._members = members

    def __repr__(self) -> str:
        return self._filename

    def _resolve(self) -> Path:
        """Fetch the dataset (downloading on first use) and return its local path."""
        if self._members is None:
            return Path(fetch(self._filename))

        paths = [fetch(f"{self._filename}{member}") for member in self._members]
        return Path(paths[0]).parent

    def graph(self) -> nx.Graph:
        return self._loader(self._resolve())

    def graphpair(self) -> libam.GraphPair:
        return self._parser(self._loader(self._resolve()))