import scipy.optimize
import networkx as nx
from pathlib import Path

from graphalign import GraphPair, eval_align
from graphalign import algorithms as alg

DATA_DIR = Path(__file__).parent.parent / "data"


def load_edgelist(path: Path) -> nx.Graph:
    """Load an undirected graph from an edge-list file with contiguous 0..n-1 labels."""
    G = nx.read_edgelist(path, nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.convert_node_labels_to_integers(G, first_label=0, ordering="sorted")


def main() -> None:
    base_graph = load_edgelist(DATA_DIR / "bio-celegans.txt")
    print(f"Base graph: {base_graph.number_of_nodes()} nodes, {base_graph.number_of_edges()} edges")

    pair = GraphPair.from_graph(base_graph).permute().add_noise(target_noise=0.05)
    print(f"Source edges: {pair.src.number_of_edges()}, Target edges: {pair.tar.number_of_edges()}")

    algorithm = alg.gradp(pair)
    print(f"Running {algorithm.name}...")
    P = algorithm.evaluate()

    ma, mb = scipy.optimize.linear_sum_assignment(-P)
    _, acc, _ = eval_align(ma, mb, pair.ground_truth[0])
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
