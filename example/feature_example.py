from pathlib import Path

from graphalign import load_alpine
from graphalign import algorithms as alg
from graphalign.evaluation.hungarian import total_eval

DATA_DIR = Path(__file__).parent.parent / "data"

def main() -> None:
    # Step 1: Load graph pair from dataset (Alpine data loading in this case)
    pair = load_alpine(DATA_DIR / "douban", "douban")
    print(f"Source: {pair.src.number_of_nodes()} nodes, {pair.src.number_of_edges()} edges")
    print(f"Target: {pair.tar.number_of_nodes()} nodes, {pair.tar.number_of_edges()} edges")

    # Step 2: Init algorithm
    algorithm = alg.gradp(pair)

    # Step 3: Run and analyze accuracy
    result = algorithm.evaluate()
    accuracy = total_eval(pair, result)
    print(f"result {algorithm.name} had a accuracy of: {accuracy:.4f}")


if __name__ == "__main__":
    main()