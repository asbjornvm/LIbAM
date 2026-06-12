Real Data
=========

The :doc:`quickstart </quickstart>` builds a graph pair from a synthetic graph. This
example does the same end-to-end pipeline on a real, bundled dataset and shows both an
unrestricted run and a restricted (anchor-based) run.

Loading a bundled dataset
-------------------------

Datasets are accessed straight from ``libam.datasets`` and downloaded on first use (see
:doc:`/api/datasets`). A single-graph dataset has identical source and target, so we
:meth:`~libam.graph.GraphPair.permute` it and add noise to create a non-trivial benchmark
with known ground truth:

.. code-block:: python

    import libam
    from libam import algorithms as alg
    from libam import datasets

    pair = datasets.bio_celegans.graphpair().permute().add_noise(target_noise=0.05)
    print(f"Source edges: {pair.src.number_of_edges()}, "
          f"Target edges: {pair.tar.number_of_edges()}")

Comparing several algorithms
----------------------------

Every algorithm takes the same pair and exposes ``align()``. Running a few and scoring
them with :func:`~libam.evaluation.evaluation.accuracy` gives a quick comparison. Sharing
one parameter dictionary across algorithms is convenient; unused keys are ignored (silence
the resulting warnings by raising the ``libam`` log level):

.. code-block:: python

    import logging
    logging.getLogger("libam").setLevel(logging.ERROR)

    params = {
        "iterations": 1, "simple": True, "mu": 0.05, "efn": 3,  # fugal
        "eta": 0.2, "init_sim": 1, "eig_type": 0,               # grampa_s
        "maxiter": 20, "alpha": 0.85,                           # isorank
    }

    algorithms = [
        alg.fugal(pair, **params),
        alg.grampa_s(pair, **params),
        alg.isorank(pair, **params),
    ]

    for algorithm in algorithms:
        P = algorithm.align()
        score = libam.evaluation.accuracy(pair, P)
        print(f"{algorithm.name}: accuracy {score:.4f}")

Using anchors with a restricted algorithm
-----------------------------------------

Restricted algorithms take a set of known correspondences. Sample them from the ground
truth with :meth:`~libam.graph.GraphPair.get_anchor_links` and pass them as
``anchor_links``:

.. code-block:: python

    anchor_links = pair.get_anchor_links(0.1)  # 10% of nodes as anchors

    algorithm = alg.joena(pair, anchor_links=anchor_links)
    P = algorithm.align()
    print(f"{algorithm.name}: accuracy {libam.evaluation.accuracy(pair, P):.4f}")