Quickstart
==========

This guide walks through the full pipeline: generating a graph pair,
running an alignment algorithm, and evaluating the result.

Installation
------------

.. code-block:: bash

    pip install graphalign

Basic Example
-------------

.. code-block:: python

    import networkx as nx

    from graphalign import GraphPair
    from graphalign import algorithms as alg
    from graphalign.evaluation.hungarian import total_eval

    # Step 1: Generate base graph
    base_graph = nx.barabasi_albert_graph(128, 4)
    print(f"Base graph: {base_graph.number_of_nodes()} nodes, " +
        f"{base_graph.number_of_edges()} edges")

    # Step 2: Build graph pair, permutes node labels and adds noise.
    pair = GraphPair.from_graph(base_graph)
                    .permute()
                    .add_noise(target_noise=0.05)
    print(f"Source edges: {pair.src.number_of_edges()}, " +
        f"Target edges: {pair.tar.number_of_edges()}")

    # Step 3: Construct algorithm parameter object and algorithm object
    parameters = {
        "iterations": 1,
        "simple": True,
        "mu": 0.05,
        "efn": 3,
    }
    algorithm = alg.fugal(pair, **parameters)

    # Step 4: Run and analyze accuracy
    result = algorithm.evaluate()
    accuracy = total_eval(pair, result)
    print(f"result {algorithm.name} had a accuracy of: {accuracy:.4f}")
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