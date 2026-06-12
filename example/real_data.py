import libam
from libam import algorithms as alg
from libam import datasets


def main() -> None:
    # Step 1: Load pair from datasets.
    pair = datasets.cora.graphpair()
    print(f"Source: {pair.src.number_of_nodes()} nodes, {pair.src.number_of_edges()} edges")
    print(f"Target: {pair.tar.number_of_nodes()} nodes, {pair.tar.number_of_edges()} edges")
    has_features = pair.src_features is not None
    print(f"Node features available: {has_features}")

    # Step 3: Remove any label-based shortcuts.
    # For two distinct graphs shuffle_labels() is the correct command, permute() is only for synthetic pairs
    pair.shuffle_labels()

    # Step 4: Sample anchors from the ground truth and run an algorithm.
    anchor_links = pair.get_anchor_links(0.1)
    algorithm = alg.joena(pair, anchor_links=anchor_links)

    result = algorithm.align()
    acc = libam.evaluation.accuracy(pair, result)
    print(f"\n{algorithm.name} had an accuracy of: {acc:.4f}")


if __name__ == "__main__":
    main()