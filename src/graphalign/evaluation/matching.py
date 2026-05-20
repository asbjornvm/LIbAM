from . import quadtree
from ..algorithms import bipartitewrapper as bmw
import numpy as np
import scipy
try:
    import lapjv
except:
    pass


def colmin(matrix):

    ma = np.arange(matrix.shape[0])

    mb = matrix.argmin(1).flatten()

    return ma, mb

def sort_greedy_voting(match_freq):
    dist_platt=np.ndarray.flatten(match_freq)
    idx = np.argsort(dist_platt)#
    n = match_freq.shape[0]
    k=idx//n
    r=idx%n
    idx_matr=np.c_[k,r]
    G1_elements=set()
    G2_elements=set()
    i= n**2 - 1
    j= 0
    matching=np.ones([n,2])*(n+1)
    while(len(G1_elements)<n):
        if (idx_matr[i,0] not in G1_elements) and (idx_matr[i,1] not in G2_elements):
            matching[j,:]=idx_matr[i,:]

            G1_elements.add(idx_matr[i,0])
            G2_elements.add(idx_matr[i,1])
            j+=1


        i-=1

    print(matching)
    print(matching[:,0])
    print(matching[:,1])
    matching = np.c_[matching[:,0], matching[:,1]]
    real_matching = dict(matching[matching[:, 0].argsort()])
    matching=matching[matching[:, 0].argsort()]
    matching.astype(int).T
    return matching


def superfast(l2):
    n = l2.shape[0]
    ma = np.zeros(n, int)
    mb = np.zeros(n, int)

    rows = set()
    cols = set()

    vals = np.argsort(l2, axis=None)

    for val in vals:

        x, y = np.unravel_index(val, l2.shape)

        if x in rows or y in cols:

            continue

        ma[x] = x
        mb[x] = y

        rows.add(x)
        cols.add(y)

    return ma, mb


def jv(dist):
    n = dist.shape[0]
    cols, rows, _ = lapjv.lapjv(dist)

    matching = np.c_[rows, np.linspace(0, n-1, n).astype(int)]
    matching = matching[matching[:, 0].argsort()]

    return matching.astype(int).T

def jv1(dist):
    n = dist.shape[0]
    try:

        cols, rows, _ = lapjv.lapjv(dist)
        matching = np.c_[np.linspace(0, n-1, n).astype(int),rows]
    except Exception:

        cols, rows = scipy.optimize.linear_sum_assignment(dist)
        matching = np.c_[rows,cols]

    matching = matching[matching[:, 0].argsort()]

    return matching.astype(int).T


def getmatching(sim, cost, mt:int, _log):
    if not isinstance(mt, int):
        raise Exception(f"mt must be a integer, but is {type(mt).__name__}")
    _log.debug("matching type: %s", mt)

    if mt > 0:
        if sim is None:
            raise Exception("Empty sim matrix")
        if mt==4:
            return np.arange(0,len(sim)),sim
        if mt == 30:
            mat_to_min = -np.log(sim)
        
        else:
            mat_to_min = sim
            mat_to_min *= -1

    else:
        if cost is None:
            raise Exception("Empty cost matrix")

        if mt == -30:
            mat_to_min = -np.log(cost)

        else:
            mat_to_min = cost

    mt = abs(mt)

    if mt == 1:
        return colmin(mat_to_min)

    elif mt == 2:
        n = mat_to_min.shape[0]

        if (n & (n-1) == 0) and n != 0:  # if n is a power of 2
            _log.debug("binary! speeding up..")
            mat_to_min *= -1

            return quadtree.superfast_binbin_torch(mat_to_min)
        else:
            return superfast(mat_to_min)

    elif mt == 3:
        try:
            return jv(mat_to_min)
        except Exception:
            return scipy.optimize.linear_sum_assignment(mat_to_min)
    elif mt == 30:
        try:
            return jv(mat_to_min)

        except Exception:
            return scipy.optimize.linear_sum_assignment(mat_to_min)

    elif mt == 98:
        return scipy.sparse.csgraph.min_weight_full_bipartite_matching(mat_to_min)

    elif mt == 99:
        return bmw.getmatchings(-mat_to_min)
    elif mt==97:
        return sort_greedy_voting(mat_to_min)
    elif mt==96:
        return jv1(mat_to_min)

    raise Exception("wrong matching config")