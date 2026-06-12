"""Functions for projecting matrix X to 0-1 permutation matrix P"""
import logging
from queue import PriorityQueue
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from tqdm import tqdm
from typing import Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from multiprocessing import shared_memory
from itertools import product
import heapq
import time
def Hungarian_all(M: Union[torch.Tensor , np.ndarray],
                  k: Union[int , None], connected: bool,
                  G1: nx.Graph, G2: nx.Graph,
                  use_torch: bool = False):
    if connected:
        k = k if k is not None else G1.number_of_nodes()
        return Hungarian_k_connected6(M, k, G1, G2, use_torch)
    if k is not None:
        return Hungarian_k(M, k, use_torch)
    return Hungarian(M, use_torch)


def Hungarian(M: Union[torch.Tensor , np.ndarray], use_torch: bool = False):
    n1, n2 = M.shape[0], M.shape[1]
    row_ind, col_ind = linear_sum_assignment(M, maximize=True)
    n = len(M)
    _type_ = torch if use_torch else np
    P = _type_.zeros((n1, n2), dtype=_type_.float64)
    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= n1) or (col_ind[i] >= n2):
            continue
    return P, row_ind, col_ind


def Hungarian_k(M: Union[torch.Tensor , np.ndarray], k: int, use_torch: bool = False):
    n1, n2 = M.shape[0], M.shape[1]
    assert k <= min(n1, n2), "k should be smaller than min(n1, n2)"
    lb = M.min()-1
    ub = M.max()+1
    M1 = np.hstack((M, ub*np.ones((n1, n1-k))))
    M2 = np.hstack((ub*np.ones((n2-k, n2)),
                    lb*np.ones((n2-k, n1-k))))
    row_ind, col_ind = linear_sum_assignment(np.vstack((M1, M2)),
                                             maximize=True)
    _type_ = torch if use_torch else np
    P = _type_.zeros((n1+n2-k, n1+n2-k), dtype=_type_.float64)
    row_ind_ret = []
    col_ind_ret = []
    counter = 0
    for i in range(n1+n2-k):
        r, c = row_ind[i], col_ind[i]
        if r < n1 and c < n2:
            counter += 1
            P[r, c] = 1
            row_ind_ret.append(r)
            col_ind_ret.append(c)
    if counter != k:
        raise RuntimeWarning(f'Hungarian found {counter}!={k} assignments.')
    return P[:n1, :n2], np.array(row_ind_ret), np.array(col_ind_ret)


def Hungarian_k_connected(M: Union[torch.Tensor , np.ndarray],
                          k: int,
                          G1: nx.Graph, G2: nx.Graph,
                          use_torch: bool = False,max_val=1000000):
    # Function for finding k-connected assignment give u-v.
    def _Hungarian_kcon(u, v, k, Pk, M, from_Pk=None):
        # Function for updating priority queu when u-v added.
        def _add_neighbors_to_queue1(u, v, pq,
                                    reach1, reach2,
                                    selec1, selec2):
            neighs1 = set(G1.neighbors(u))
            neighs2 = set(G2.neighbors(v))
            for neigh1 in neighs1 - reach1:     # new neigh1
                for neigh2 in ((neighs2-reach2) | (reach2 - selec2)):  # new+old neigh2
                    pq.put((-(Pk[neigh1, neigh2] + M[neigh1, neigh2]),
                           (neigh1, neigh2)))
            for neigh1 in reach1 - selec1:  # old neigh1
                for neigh2 in neighs2 - reach2:  # new neigh2
                    pq.put((-(Pk[neigh1, neigh2] + M[neigh1, neigh2]),
                           (neigh1, neigh2)))
            reach1.update(neighs1 - reach1)         
            reach2.update(neighs2 - reach2)          
        # Init structures
        def _add_neighbors_to_queue(u, v, pq,
                                    reach1, reach2,
                                    selec1, selec2):
            neighs1 = set(G1.neighbors(u))
            neighs2 = set(G2.neighbors(v))
            # to_add1 = set()
            # to_add2 = set()
            new_neigh1 = neighs1 - reach1
            new_neigh2 = neighs2 - reach2
            old_neigh1 = reach1 - selec1
            old_neigh2 = reach2 - selec2
            for neigh1 in new_neigh1:     # new neigh1
                #to_add1.add(neigh1)
                for neigh2 in ((new_neigh2) | (old_neigh2)):  # new+old neigh2
                    pq.put((-(Pk[neigh1, neigh2] + M[neigh1, neigh2]),
                           (neigh1, neigh2)))
            for neigh1 in old_neigh1:  # old neigh1
                for neigh2 in new_neigh2:  # new neigh2
                    #to_add2.add(neigh2)
                    pq.put((-(Pk[neigh1, neigh2] + M[neigh1, neigh2]),
                           (neigh1, neigh2)))
            reach1.update(new_neigh1)         
            reach2.update(new_neigh2)
        row_ind_ret = []
        col_ind_ret = []
        reach1, reach2 = set(), set()
        selec1, selec2 = set(), set()
        pq = PriorityQueue()
        pq.put((-max_val, (u, v)))
        reach1.add(u)                              
        reach2.add(v)
        _add_neighbors_to_queue(u, v, pq, reach1, reach2, selec1, selec2)
        counter = k
        score = 0
        # Iterate pairs in queue until forming a k-connected bijection
        while not pq.empty() and counter > 0:
            u, v = pq.get()[1]
            if (u in selec1) or (v in selec2):
                continue
            counter -= 1
            selec1.add(u)
            selec2.add(v)
            
            row_ind_ret.append(u)
            col_ind_ret.append(v)
            score += M[u, v]
            if from_Pk and Pk[u, v] == 1:
                from_Pk.add(f'{u}-{v}')
            _add_neighbors_to_queue(u, v, pq, reach1, reach2, selec1, selec2)
        if counter != 0:
            raise ValueError(f"Found {len(row_ind_ret)}!={k} pairs")
        return row_ind_ret, col_ind_ret, score
    # Assertion checks
    ccG1 = nx.connected_components(G1)
    ccG2 = nx.connected_components(G2)
    assert max(len(c1) for c1 in ccG1) >= k, "Maximum cc of G1 is less than k"
    assert max(len(c2) for c2 in ccG2) >= k, "Maximum cc of G2 is less than k"
    ccG1 = nx.connected_components(G1)
    ccG2 = nx.connected_components(G2)
    # Running k-size Hungarian on feasible M
    for c1 in ccG1:

        if len(c1) < k:
            M[list(c1), :] = -float(np.inf)
            continue
    for c2 in ccG2:
        if len(c2) < k:
            M[:, list(c2)] = -float(np.inf)
            continue
    # Pk, row_ind_k, col_ind_k = Hungarian(M, use_torch)
    
    Pk, row_ind_k, col_ind_k = Hungarian_k(M, k, use_torch)
    # Bijection pairs sorted by M[u,v] value
    pq_Pk: PriorityQueue = PriorityQueue()
    for i in range(len(row_ind_k)):
        u, v = row_ind_k[i], col_ind_k[i]
        pq_Pk.put((-M[u, v], (u, v)))
    # Iterate over all pairs based on their score
    row_ind_kcon, col_ind_kcon, score = np.array([]), np.array([]), -1
    from_Pk = set()
    pbar = tqdm(total=pq_Pk.qsize(),
                desc="Constructing k connected bijection")
    for i in range(pq_Pk.qsize()):
        pbar.update(1)
        u, v = pq_Pk.get()[1]
        if f'{u}-{v}' in from_Pk:
            continue
        from_Pk.add(f'{u}-{v}')
        row_ind_kcon_uv, col_ind_kcon_uv, score_uv = _Hungarian_kcon(u, v, k,
                                                                     Pk, M,
                                                                     from_Pk)
        if score_uv > score:
            score = score_uv
            row_ind_kcon, col_ind_kcon = row_ind_kcon_uv, col_ind_kcon_uv

    n1, n2 = M.shape[0], M.shape[1]
    _type_ = torch if use_torch else np
    Pkcon = _type_.zeros((n1, n2), dtype=_type_.float64)
    for i in range(k):
        Pkcon[row_ind_kcon[i]][col_ind_kcon[i]] = 1
    return Pkcon, row_ind_kcon, col_ind_kcon


def Hungarian_k_connected2(M: Union[torch.Tensor , np.ndarray],
                          k: int,
                          G1: nx.Graph, G2: nx.Graph,
                          use_torch: bool = False):
    # Function for finding k-connected assignment give u-v.
    def _Hungarian_kcon(u, v, k, Pk, Phung, M, from_Pk=None):
        # Function for updating priority queu when u-v added.
        def _add_neighbors_to_queue(u, v, pq,
                                    reach1, reach2,
                                    selec1, selec2):
            neighs1 = set(G1.neighbors(u))
            neighs2 = set(G2.neighbors(v))
            to_add1 = set()
            to_add2 = set()
            for neigh1 in neighs1 - reach1:     # new neigh1
                to_add1.add(neigh1)
                for neigh2 in neighs2.union(reach2 - selec2):  # new+old neigh2
                    #pq.put((-(0.0*Phung[neigh1, neigh2] + M[neigh1, neigh2]),
                    pq.put((-(Phung[neigh1, neigh2] + M[neigh1, neigh2]),
                           (neigh1, neigh2)))
            for neigh1 in reach1 - selec1:  # old neigh1
                for neigh2 in neighs2 - reach2:  # new neigh2
                    to_add2.add(neigh2)
                    pq.put((-(Phung[neigh1, neigh2] + M[neigh1, neigh2]),
                           (neigh1, neigh2)))
            for neigh1 in to_add1:
                reach1.add(neigh1)
            for neigh2 in to_add2:
                reach2.add(neigh2)

        def _add_neighbors_to_queue1(u, v, pq,
                                    reach1, reach2,
                                    selec1, selec2):

            neighs1 = set(G1.neighbors(u)) - selec1
            neighs2 = set(G2.neighbors(v)) - selec2

            new1 = neighs1 - reach1
            new2 = neighs2 - reach2

            all2 = reach2 | new2

            for n1 in new1:
                for n2 in all2:
                    pq.put((-(Phung[n1, n2] + M[n1, n2]), (n1, n2)))

            for n1 in reach1 - selec1:
                for n2 in new2:
                    pq.put((-(Phung[n1, n2] + M[n1, n2]), (n1, n2)))

            reach1.update(new1)
            reach2.update(new2)
        # Init structures
        row_ind_ret = []
        col_ind_ret = []
        reach1, reach2 = set(), set()
        selec1, selec2 = set(), set()
        pq = PriorityQueue()
        pq.put((-np.inf, (u, v)))
        _add_neighbors_to_queue(u, v, pq, reach1, reach2, selec1, selec2)
        counter = k
        score = 0
        # Iterate pairs in queue until forming a k-connected bijection
        while not pq.empty() and counter > 0:
            u, v = pq.get()[1]
            if (u in selec1) or (v in selec2):
                continue
            counter -= 1
            selec1.add(u)
            selec2.add(v)
            row_ind_ret.append(u)
            col_ind_ret.append(v)
            score += M[u, v]
            if from_Pk and Pk[u, v] == 1:
                from_Pk.add(f'{u}-{v}')
            _add_neighbors_to_queue1(u, v, pq, reach1, reach2, selec1, selec2)
        if counter != 0:
            raise ValueError(f"Found {len(row_ind_ret)}!={k} pairs")
        return row_ind_ret, col_ind_ret, score
    # Assertion checks
    ccG1 = nx.connected_components(G1)
    ccG2 = nx.connected_components(G2)
    assert max(len(c1) for c1 in ccG1) >= k, "Maximum cc of G1 is less than k"
    assert max(len(c2) for c2 in ccG2) >= k, "Maximum cc of G2 is less than k"
    ccG1 = nx.connected_components(G1)
    ccG2 = nx.connected_components(G2)
    # Running k-size Hungarian on feasible M
    penalty= -(torch.max(M)*k+ 1)

    for c1 in ccG1:
        if len(c1) < k:
            M[list(c1), :] = float(penalty)
            continue
    for c2 in ccG2:
        if len(c2) < k:
            M[:, list(c2)] = float(penalty)
            continue
    Phung, _, _ = Hungarian(M, use_torch)
    Pk, row_ind_k, col_ind_k = Hungarian_k(M, k, use_torch)
    # Bijection pairs sorted by M[u,v] value
    pq_Pk: PriorityQueue = PriorityQueue()
    for i in range(len(row_ind_k)):
        u, v = row_ind_k[i], col_ind_k[i]
        pq_Pk.put((-M[u, v], (u, v)))
    # Iterate over all pairs based on their score
    row_ind_kcon, col_ind_kcon, score = np.array([]), np.array([]), -1
    from_Pk = set()
    pbar = tqdm(total=pq_Pk.qsize(),
                desc="Constructing k connected bijection")
    for i in range(pq_Pk.qsize()):
        pbar.update(1)
        u, v = pq_Pk.get()[1]
        if f'{u}-{v}' in from_Pk:
            continue
        from_Pk.add(f'{u}-{v}')
        row_ind_kcon_uv, col_ind_kcon_uv, score_uv = _Hungarian_kcon(u, v, k,
                                                                     Pk, Phung, M,
                                                                     from_Pk)
        if score_uv > score:
            score = score_uv
            row_ind_kcon, col_ind_kcon = row_ind_kcon_uv, col_ind_kcon_uv

    n1, n2 = M.shape[0], M.shape[1]
    _type_ = torch if use_torch else np
    Pkcon = _type_.zeros((n1, n2), dtype=_type_.float64)
    for i in range(k):
        Pkcon[row_ind_kcon[i]][col_ind_kcon[i]] = 1
    ones_count = int(Pkcon.sum())
    row_counts = {}
    col_counts = {}
    for r, c in zip(row_ind_kcon, col_ind_kcon):
        row_counts[r] = row_counts.get(r, 0) + 1
        col_counts[c] = col_counts.get(c, 0) + 1

    duplicate_rows = {node: cnt for node, cnt in row_counts.items() if cnt > 1}
    duplicate_cols = {node: cnt for node, cnt in col_counts.items() if cnt > 1}

    if duplicate_rows:
        raise ValueError(f"G1 nodes selected more than once: {duplicate_rows}")
    if duplicate_cols:
        raise ValueError(f"G2 nodes selected more than once: {duplicate_cols}")

    logging.getLogger(__name__).info(f"All {k} nodes in G1 and G2 selected exactly once. OK.")
    return Pkcon, row_ind_kcon, col_ind_kcon



def Hungarian_k_connected6(M: Union[torch.Tensor , np.ndarray],
                          k: int,
                          G1: nx.Graph, G2: nx.Graph,
                          use_torch: bool = False):
    
    # 1. Pre-convert graphs to adjacency sets for O(1) lookups
    n1_nodes = G1.number_of_nodes()
    n2_nodes = G2.number_of_nodes()
    adj1 = [set(G1.neighbors(i)) for i in range(n1_nodes)]
    adj2 = [set(G2.neighbors(i)) for i in range(n2_nodes)]
    
    if torch.is_tensor(M):
        M_np = M.detach().cpu().numpy()
    else:
        M_np = M.copy()

    # --- Main Logic ---
    ccG1 = list(nx.connected_components(G1))
    ccG2 = list(nx.connected_components(G2))
    
    assert max(len(c) for c in ccG1) >= k, "Maximum cc of G1 is less than k"
    assert max(len(c) for c in ccG2) >= k, "Maximum cc of G2 is less than k"

    max_M = np.max(M_np)
    penalty = -(k * max_M + 1)
    
    # Penalize components smaller than k
    for c1 in ccG1:
        if len(c1) < k:
            M_np[list(c1), :] = penalty
    for c2 in ccG2:
        if len(c2) < k:
            M_np[:, list(c2)] = penalty

    # Assumes Hungarian_k is defined in scope
    Pk, row_ind_k, col_ind_k = Hungarian_k(M_np, k, use_torch)
    #Phung, _, _ = Hungarian_k(M_np, M_np.shape[0], use_torch)
    Phung, _, _ = Hungarian(M, use_torch)
    # Ensure matrices are numpy to avoid slow torch tensor lookups in loops
    Pk_np = Pk.detach().cpu().numpy() if torch.is_tensor(Pk) else Pk
    Phung_np = Phung.detach().cpu().numpy() if torch.is_tensor(Phung) else Phung

    # 2. Precompute the Weight Matrix W ONCE outside the loop
    Pall = 2.0 * Phung_np + 4.0 * Pk_np
    # Pall = 1.0 * Pk_np
    W = Pall + M_np
    
    # 3. Precompute valid Pk pairs into a set for fast O(1) lookups
    Pk_pairs = set(zip(*np.where(Pk_np > 0.5)))

    # Define inner functions outside the loop so they aren't re-compiled repeatedly
    def _add_neighbors_to_queue(u, v, pq, reach1, reach2, selec1, selec2):
        newneigh1 = adj1[u] - reach1
        newneigh2 = adj2[v] - reach2
        
        reach1.update(newneigh1)
        reach2.update(newneigh2)
        
        # 4. Prune pairs: Do not push pairs involving already selected nodes
        valid_new1 = newneigh1 - selec1
        valid_new2 = newneigh2 - selec2
        valid_reach1 = reach1 - selec1
        valid_reach2 = reach2 - selec2

        # 1. New neighbors of u paired with all valid reachable neighbors of v
        if valid_new1:
            for n1, n2 in product(valid_new1, valid_reach2):
                # 5. Flatten the tuple to avoid inner tuple creation overhead
                heapq.heappush(pq, (-W[n1, n2], n1, n2))
        
        # 2. Existing reachable neighbors of u paired with NEW neighbors of v
        if valid_new2:
            # valid_reach1 already contains valid_new1. Prevent duplicate pushes:
            old_valid_reach1 = valid_reach1 - valid_new1
            if old_valid_reach1:
                for n1, n2 in product(old_valid_reach1, valid_new2):
                    heapq.heappush(pq, (-W[n1, n2], n1, n2))

    def _Hungarian_kcon(u, v, max_val):
        row_ind_ret, col_ind_ret = [], []
        reach1, reach2 = {u}, {v}
        selec1, selec2 = set(), set()
        
        pq = []
        heapq.heappush(pq, (-max_val, u, v))
        
        _add_neighbors_to_queue(u, v, pq, reach1, reach2, selec1, selec2)
        
        counter = k
        score = 0
        while pq and counter > 0:
            priority, curr_u, curr_v = heapq.heappop(pq)
            if (curr_u in selec1) or (curr_v in selec2):
                continue
            counter -= 1
            selec1.add(curr_u)
            selec2.add(curr_v)
            row_ind_ret.append(curr_u)
            col_ind_ret.append(curr_v)
            score += M_np[curr_u, curr_v]
            
            # Use the O(1) set lookup instead of matrix indexing
            if (curr_u, curr_v) in Pk_pairs:
                from_Pk.add((curr_u, curr_v))
            
            _add_neighbors_to_queue(curr_u, curr_v, pq, reach1, reach2, selec1, selec2)            
        if counter != 0:
            raise ValueError(f"Found {len(row_ind_ret)}!={k} pairs")
        return row_ind_ret, col_ind_ret, score

    # Queue of initial starting pairs based on Pk
    pq_Pk = []
    for i in range(len(row_ind_k)):
        u, v = row_ind_k[i], col_ind_k[i]
        heapq.heappush(pq_Pk, (-M_np[u, v], u, v))

    row_ind_kcon, col_ind_kcon, score = [], [], -1
    from_Pk = set()
    max_val = (k * max_M + 1)
    counter = 0
    time_start=time.time()
    while pq_Pk:
        priority, u, v = heapq.heappop(pq_Pk)
        counter+=1
        if (u, v) in from_Pk:
            continue
        
        
        from_Pk.add((u, v))
        row_ind_kcon_uv, col_ind_kcon_uv, score_uv = _Hungarian_kcon(u, v, max_val)
        
        if score_uv > score:
            score = score_uv
            row_ind_kcon, col_ind_kcon = row_ind_kcon_uv, col_ind_kcon_uv
        time_end=time.time()    
        if (time_end-time_start>3600):
            break
    n1, n2 = M_np.shape
    if use_torch:
        Pkcon = torch.zeros((n1, n2), dtype=torch.float64)
    else:
        Pkcon = np.zeros((n1, n2), dtype=np.float64)
    
    # 6. Use advanced vectorized indexing instead of a for loop
    if len(row_ind_kcon) > 0:
        Pkcon[row_ind_kcon, col_ind_kcon] = 1.0
        
    return Pkcon, row_ind_kcon, col_ind_kcon       