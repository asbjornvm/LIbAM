import libam.evaluation
import libam.datasets as datasets
from libam import algorithms as alg


def main() -> None:
    pair = datasets.bio_celegans.graphpair().permute().add_noise(target_noise=0.05)
    print(f"Source edges: {pair.src.number_of_edges()}, Target edges: {pair.tar.number_of_edges()}")

    algorithm = alg.gradp(pair)
    print(f"Running {algorithm.name}...")

    p = algorithm.align()
    acc = libam.evaluation.accuracy(pair, p)
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
