Algorithms
==========

All algorithms take a :class:`~libam.graph.GraphPair` and return an
instance with an ``align()`` method that produces a similarity matrix often referred to as ``P``.

The similarity matrix is an ``n * n`` float array where ``P[i, j]`` scores
how likely source node ``i`` maps to target node ``j``.
This matrix is doubly stochastic.

To turn ``P`` into a concrete node-to-node alignment and score it, pass the pair and ``P``
to the :doc:`evaluation functions <evaluation>`. They resolve the optimal assignment
internally (via the Hungarian algorithm on ``-P``) so you do not have to extract it by hand:

.. code-block:: python

    from libam import evaluation
    score = evaluation.accuracy(pair, P)

Note that this is not built for scale, just for demonstrating how evaluation can be done.

Algorithms fall into two groups. **Unrestricted** algorithms align the graphs from
structure (and optional features) alone. **Restricted** algorithms additionally use
node data in order to align nodes with more detailed information.

Each factory below takes a :class:`~libam.graph.GraphPair` and returns an algorithm
instance whose ``align()`` method produces ``P``.

Unrestricted
------------

.. autofunction:: libam.algorithms.fugal

.. autofunction:: libam.algorithms.grampa

.. autofunction:: libam.algorithms.grampa_s

.. autofunction:: libam.algorithms.cone

.. autofunction:: libam.algorithms.gwl

.. autofunction:: libam.algorithms.sgwl

.. autofunction:: libam.algorithms.path

.. autofunction:: libam.algorithms.isorank

.. autofunction:: libam.algorithms.nsd

.. autofunction:: libam.algorithms.mmnc

.. autofunction:: libam.algorithms.dspp

.. autofunction:: libam.algorithms.fgot

.. autofunction:: libam.algorithms.got

.. autofunction:: libam.algorithms.grasp

.. autofunction:: libam.algorithms.klaus

.. autofunction:: libam.algorithms.lera

.. autofunction:: libam.algorithms.mds

.. autofunction:: libam.algorithms.net_align

.. autofunction:: libam.algorithms.parrot

.. autofunction:: libam.algorithms.regal

Restricted
----------

.. autofunction:: libam.algorithms.gradp

.. autofunction:: libam.algorithms.joena

.. autofunction:: libam.algorithms.htc

.. autofunction:: libam.algorithms.slot_a

.. autofunction:: libam.algorithms.alpine

Base Class
----------

.. autoclass:: libam.algorithms.algorithm.AlignAlgorithm
   :members: