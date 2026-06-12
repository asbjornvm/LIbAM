import networkx as nx

import libam
from libam import GraphPair
from libam import algorithms as alg

def main() -> None:
    # Step 1: Generate base graph
    base_graph = nx.barabasi_albert_graph(128, 4)
    print(f"Base graph: {base_graph.number_of_nodes()} nodes, {base_graph.number_of_edges()} edges")

    # Step 2: Build graph pair, permutes node labels and adds noise.
    pair = GraphPair.from_graph(base_graph).permute().add_noise(target_noise=0.05)
    print(f"Source edges: {pair.src.number_of_edges()}, Target edges: {pair.tar.number_of_edges()}")

    # Step 3: Construct algorithm with parameter
    algorithm = alg.fugal(pair, 15, True, 0.05, 3)

    # Step 4: Run and analyze accuracy
    result = algorithm.align()
    acc = libam.evaluation.accuracy(pair, result)
    print(f"{algorithm.name} had an accuracy of: {acc:.4f}")


if __name__ == "__main__":
    main()