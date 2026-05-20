import networkx as nx

from graphalign import GraphPair
from graphalign import algorithms as alg
from graphalign.evaluation.hungarian import total_eval

def main() -> None:
    # Step 1: Generate base graph
    base_graph = nx.barabasi_albert_graph(128, 4)
    print(f"Base graph: {base_graph.number_of_nodes()} nodes, {base_graph.number_of_edges()} edges")

    # Step 2: Build graph pair, permutes node labels and adds noise.
    pair = GraphPair.from_graph(base_graph).permute().add_noise(target_noise=0.05)
    print(f"Source edges: {pair.src.number_of_edges()}, Target edges: {pair.tar.number_of_edges()}")

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


if __name__ == "__main__":
    main()