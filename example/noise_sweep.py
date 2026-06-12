import copy

import networkx as nx
import numpy as np

import libam
from libam import GraphPair
from libam import algorithms as alg


def main() -> None:

    # Step 1: Generate a single base graph, reused for every noise level
    base_graph = nx.barabasi_albert_graph(128, 4)
    print(f"Base graph: {base_graph.number_of_nodes()} nodes, "
          f"{base_graph.number_of_edges()} edges\n")

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    repeats = 3  # average over a few random noise draws per level to reduce variance

    # Permute once
    # deep copy to save base preset
    base_pair = GraphPair.from_graph(base_graph).permute()

    print(f"{'noise':>7} | {'accuracy':>9}")
    for noise in noise_levels:
        accuracies = []
        for _ in range(repeats):
            # Copy the clean pair, then add noise
            pair: GraphPair = copy.deepcopy(base_pair)
            pair = pair.add_noise(target_noise=noise)
            result = alg.fugal(pair, 15, True, 0.05, 3).align()
            accuracies.append(libam.evaluation.accuracy(pair, result))

        mean_acc = float(np.mean(accuracies))
        print(f"{noise:>7.2f} | {mean_acc:>9.4f}")


if __name__ == "__main__":
    main()