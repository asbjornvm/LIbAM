import networkx as nx
import numpy as np

from libam import GraphPair
from libam import algorithms as alg
from libam.evaluation import accuracy

def main() -> None:
    # Step 1: Generate base graph
    base_graph = nx.barabasi_albert_graph(128, 4)
    print(f"Base graph: {base_graph.number_of_nodes()} nodes, {base_graph.number_of_edges()} edges")

    # Step 2: Build graph pair, permutes node labels and adds noise.
    pair = GraphPair.from_graph(base_graph).permute().add_noise(target_noise=0.05)
    print(f"Source edges: {pair.src.number_of_edges()}, Target edges: {pair.tar.number_of_edges()}")

    anchor_links = pair.get_anchor_links(0.1)

    # Step 3: Construct algorithm parameter object and algorithm object
    parameters = {
        "anchor_links": anchor_links
    }

    algorithm = alg.joena(pair, **parameters)

    # Step 4: Run and analyze accuracy
    result = algorithm.align()
    acc = accuracy(pair, result)
    print(f"result {algorithm.name} had an accuracy of: {acc:.4f}")


if __name__ == "__main__":
    main()