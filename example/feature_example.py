from libam import algorithms as alg
import libam
import libam.datasets as datasets


def main() -> None:
    # Step 1: Load graph pair from dataset (Alpine data loading in this case)
    pair = datasets.inf_power.graphpair().add_noise(0.0, refill=True)
    print(f"Source: {pair.src.number_of_nodes()} nodes, {pair.src.number_of_edges()} edges")
    print(f"Target: {pair.tar.number_of_nodes()} nodes, {pair.tar.number_of_edges()} edges")

    anchor_links = pair.get_anchor_links(0.1)

    # Step 2: Init algorithm
    algorithm = alg.joena(pair, anchor_links)

    # Step 3: Run and analyze accuracy
    result = algorithm.align()
    accuracy = libam.evaluation.accuracy(pair, result)
    print(f"result {algorithm.name} had an accuracy of: {accuracy:.4f}")


if __name__ == "__main__":
    main()