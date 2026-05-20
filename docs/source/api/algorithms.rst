Algorithms
==========

All algorithms take a :class:`~graphalign.graph.GraphPair` and return an
instance with an ``evaluate()`` method that produces a similarity matrix ``P``.

The similarity matrix is an ``n * n`` float array where ``P[i, j]`` scores
how likely source node ``i`` maps to target node ``j``.

A common way of interpriting the scores to an alginment is by passing ``-P`` to
``scipy.optimize.linear_sum_assignment`` (Hungarian algorithm) in order to extract the best assignment.
Node ``ma[i]`` in source graph is matched to node ``mb[i]`` in target graph. This method will find the optimal
assignment but is not the fastest.

.. code-block:: python

    ma, mb = scipy.optimize.linear_sum_assignment(-P)

Factory Functions
-----------------
A number of

.. autofunction:: graphalign.algorithms.fugal

.. autofunction:: graphalign.algorithms.grampa

.. autofunction:: graphalign.algorithms.grampa_s

Base Class
----------

.. autoclass:: graphalign.algorithms.algorithm.Algorithm
   :members: