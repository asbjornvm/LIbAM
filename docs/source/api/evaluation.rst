Evaluation
==========

LibAM offers a small set of metrics for alignment quality. Accuracy can be used when the
ground truth between the aligned graphs is known. When it is not, Frobenius offers an
alternative based purely on how well the matched structure overlaps.

These evaluation functions are largely meant for learning your way around the library, and are
not built to scale to larger datasets. They exist as a small set of tools to test
alignments.

The primary metrics take a :class:`~libam.graph.GraphPair` and the resulting similarity
matrix ``P``, found by executing an alignment algorithm.

Primary
-------

.. autofunction:: libam.evaluation.evaluation.accuracy

.. autofunction:: libam.evaluation.evaluation.frobenius

