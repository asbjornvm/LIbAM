import numpy as np
import networkx as nx
import scipy.sparse as sps

def refill_e(edges, n, amount):
    if amount == 0:
        return edges

    ee = {tuple(row) for row in np.sort(edges).tolist()}
    new_e = []
    check = 0
    while len(new_e) < amount:
        _e = np.random.randint(n, size=2)
        _ee = tuple(np.sort(_e).tolist())
        check += 1
        if _ee not in ee and _e[0] != _e[1]:
            ee.add(_ee)
            new_e.append(_e)
            check = 0
        if check % 1000 == 999:
            print(f"refill - {check + 1} times in a row fail")
    # print(new_e)
    return np.append(edges, new_e, axis=0)


def remove_e(edges, noise, no_disc=True, until_connected=False):
    ii = 0
    while True:
        ii += 1

        if no_disc:
            bin_count = np.bincount(edges.flatten())
            rows_to_delete = []
            for i, edge in enumerate(edges):
                if np.random.sample(1)[0] < noise:
                    e, f = edge
                    if bin_count[e] > 1 and bin_count[f] > 1:
                        bin_count[e] -= 1
                        bin_count[f] -= 1
                        rows_to_delete.append(i)
            new_edges = np.delete(edges, rows_to_delete, axis=0)
        else:
            new_edges = edges[np.random.sample(edges.shape[0]) >= noise]

        graph = nx.Graph(new_edges.tolist())
        graph_cc = len(max(nx.connected_components(graph), key=len))
        graph_connected = graph_cc == np.amax(edges) + 1
        if graph_connected or not until_connected:
            break
    return new_edges


def load_as_nx(path):
    G_e = np.loadtxt(path, int)
    G = nx.Graph(G_e.tolist())
    print("Just checking",nx.is_directed(G))
    return np.array(G.edges)


def permute_graph(src_edges: np.ndarray, n: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Apply a random node-index permutation to src_edges, producing a target edge list.

    Returns:
        tar_edges: edge list with permuted node labels
        Gt: ground truth tuple (src_to_tar, tar_to_src) mappings
    """
    ground_truth_edges = np.array((
        np.arange(n),
        np.random.permutation(n),
    ))
    ground_truth = (
        ground_truth_edges[:, ground_truth_edges[1].argsort()][0],
        ground_truth_edges[:, ground_truth_edges[0].argsort()][1],
    )
    tar_edges = ground_truth[0][src_edges]
    return tar_edges, ground_truth


def apply_noise(
        src_edges: np.ndarray, tar_edges: np.ndarray,
        n: int, n_edges: int,
        source_noise: float = 0.0, target_noise: float = 0.0,
        refill: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Remove (and optionally refill) edges from source and/or target edge lists.

    Returns:
        (src_edges, tar_edges) with noise applied
    """
    src_edges = remove_e(src_edges, source_noise)
    tar_edges = remove_e(tar_edges, target_noise)

    if refill:
        src_edges = refill_e(src_edges, n, n_edges - src_edges.shape[0])
        tar_edges = refill_e(tar_edges, n, n_edges - tar_edges.shape[0])

    return src_edges, tar_edges


# TODO: Remove?
def generate_graphs(G, source_noise=0.00, target_noise=0.00, refill=False):
    if isinstance(G, list):
        print("A")
        _src, _tar, _gt = G

        Src_e = load_as_nx(_src)
        Tar_e = load_as_nx(_tar)

        if isinstance(_gt, str):
            gt_e = np.loadtxt(_gt, int).T
        elif _gt is None:
            gt1 = np.arange(
                max(np.amax(Src_e), np.amax(Tar_e))+1).reshape(-1, 1)
            gt_e = np.repeat(gt1, 2, axis=1).T
            print(gt_e)
        else:
            gt_e = np.array(_gt, int)

        #Gt = (
        #    gt_e[:, gt_e[1].argsort()][0],
        #    gt_e[:, gt_e[0].argsort()][1]
        #)
        n = np.amax(Tar_e) + 1
        nedges =Tar_e.shape[0]

        gt_e = np.array((
            np.arange(n),
            np.random.permutation(n)
        ))
        Gt = (
        gt_e[:, gt_e[1].argsort()][0],
        gt_e[:, gt_e[0].argsort()][1]
        )
        Tar_e1 = Gt[0][Tar_e]
        return Src_e, Tar_e1, Gt

    elif isinstance(G, str):
        Src_e = load_as_nx(G)
        print("B")
    elif isinstance(G, nx.Graph):
        Src_e = np.array(G.edges)
        print("C")
    else:
        print("D")
        return sps.csr_matrix([]), sps.csr_matrix([]), (np.empty(1), np.empty(1))
    if (np.amin(Src_e)!=0):
        print("E")
        Src_e=Src_e-np.amin(Src_e)
    n = np.amax(Src_e) + 1
    nedges = Src_e.shape[0]

    Tar_e, Gt = permute_graph(Src_e, n)
    Src_e, Tar_e = apply_noise(Src_e, Tar_e, n, nedges, source_noise, target_noise, refill)

    return Src_e, Tar_e, Gt
