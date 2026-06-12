Quickstart
==========

This guide walks through the full pipeline: generating a graph pair,
running an alignment algorithm, and evaluating the result.

Installation
------------
Use pip to install the library

.. code-block:: bash

    pip install libam

Basic Example
-------------

.. code-block:: python

    import networkx as nx

    from libam import GraphPair
    from libam import algorithms as alg
    from libam.evaluation.hungarian import total_eval

    # Step 1: Generate base graph
    base_graph = nx.barabasi_albert_graph(128, 4)
    print(f"Base graph: {base_graph.number_of_nodes()} nodes, {base_graph.number_of_edges()} edges")

    # Step 2: Build graph pair, permutes node labels and adds noise.
    pair = GraphPair.from_graph(base_graph).permute().add_noise(target_noise=0.05)
    print(f"Source edges: {pair.src.number_of_edges()}, Target edges: {pair.tar.number_of_edges()}")

    # Step 3: Construct algorithm parameter object and algorithm object
    parameters = {
        "alg_a": {
            "iterations": 1,
            "simple": True,
            "mu": 0.05,
            "efn": 3,
        },
        "alg_b": {
            "iters": 30,
            "method": "lowrank_svd_union",
            "b_match": 1,
            "default_params": True
        }
    }
    algorithm_a = alg.fugal(pair, **parameters["alg_a"])
    algorithm_b = alg.lera(pair, **parameters["alg_b"])

    # Step 4: Run and analyze accuracy
    result_a, result_b = algorithm_a.evaluate(), algorithm_b.evaluate()
    accuracy_a, accuracy_b = total_eval(pair, result_a), total_eval(pair, result_b)
    print(f"result {algorithm_a.name} had an accuracy of: {accuracy_a:.4f}")
    print(f"result {algorithm_b.name} had an accuracy of: {accuracy_b:.4f}")

Controlling the Graph Pair
--------------------------

By default, ``GraphPair.from_graph`` permutes node labels and optionally
removes edges from either graph. The permutation is always applied, it is
what makes the alignment problem non-trivial.

.. code-block:: python

    # Only target gets noise (default use case)
    pair = GraphPair.from_graph(G, target_noise=0.05)

    # Both graphs get noise
    pair = GraphPair.from_graph(G, source_noise=0.05, target_noise=0.05)

    # Remove edges but refill with random ones to preserve edge count
    pair = GraphPair.from_graph(G, target_noise=0.10, refill=True)

    # Permutation only, no noise
    pair = GraphPair.from_graph(G)

For advanced use, the permutation and noise steps are also available
as standalone functions:

Using an Existing Graph Pair
-----------------------------

If you already have two separate graphs, use ``from_graphs``:

.. code-block:: python

    pair = GraphPair.from_graphs(src_graph, tar_graph, ground_truth=gt)