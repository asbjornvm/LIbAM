import numpy as np

import libam.datasets
from libam import algorithms as alg
import libam


def main() -> None:
    # Step 0: Disable warning logging of you know what you are doing,
    # can be useful to clean up messy multi algorithm unused parameter warning noise
    import logging
    logging.getLogger("libam").setLevel(logging.ERROR)

    # Step 1: Load graph pair, and permute + noise data
    pair = libam.datasets.bio_dmela.graphpair().permute().add_noise(target_noise=0.05)
    print(f"Source edges: {pair.src.number_of_edges()}, Target edges: {pair.tar.number_of_edges()}")

    # Step 3: Construct shared params object, and set up algorithms
    params = {
        # Fugal parts
        "iterations": 1,
        "simple": True,
        "mu": 0.05,
        "efn": 3,
        # GrampaS parts
        "eta": 0.2,
        "init_sim": 1,
        "eig_type": 0,
        # Isorank
        "maxiter": 20,
        "alpha": 0.85,
    }

    # If logging level is not set right, using multiple algorithms will
    # likely produce warnings as they will be provided unused parameters
    algorithms = [
        alg.fugal(pair, **params),
        alg.grampa_s(pair, **params),
        alg.isorank(pair, **params),
    ]

    # Step 4: Run algorithm
    results: dict[str, np.ndarray] = {} # A collection of algorithm run identifiers and their matrix result

    for algorithm in algorithms:
        print(f"Running {algorithm.name}...")
        results[algorithm.name] = algorithm.align()

    # Step 5: Analyze accuracy
    accuracy = [(libam.evaluation.accuracy(pair, permutation)) for name, permutation in results.items()]
    frobenius = [(libam.evaluation.frobenius(pair, permutation)) for name, permutation in results.items()]
    for i in range(len(algorithms)):
        print(f"result {algorithms[i].name} had an accuracy of: {accuracy[i]:.4f}, with Frobenius score: {str(frobenius[i])}")


if __name__ == "__main__":
    main()