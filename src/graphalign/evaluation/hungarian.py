import numpy as np
import scipy

from ..graph import GraphPair
from . import eval_align

def total_eval(pair:GraphPair, p: np.ndarray) -> float:
    # default Hungarian matching, find the best
    # node-to-node assignment.
    ma, mb = scipy.optimize.linear_sum_assignment(-p)

    _, acc, _ = eval_align(ma, mb, pair.ground_truth[0])
    return  acc

def multi_eval(pair: GraphPair, eval_list: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    return [(name, total_eval(pair, permutation)) for name, permutation in eval_list.items()]