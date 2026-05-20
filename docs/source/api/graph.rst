Graph
=====

The ``graph`` module provides :class:`~graphalign.graph.GraphPair`, the central
data object of the library. It holds a source and target graph and exposes
all derived representations, adjacency matrices, node-pair candidates,
edge-pair adjacency.

Algorithms access whatever format they need directly from the pair.
Nothing is computed until first accessed, and each representation is
computed at most once.

.. autoclass:: graphalign.graph.GraphPair
   :members:
   :undoc-members: