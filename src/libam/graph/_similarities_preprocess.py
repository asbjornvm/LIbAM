from math import log2, floor
import numpy as np
import scipy.sparse as sps


def create_S(A, B, L):
    n = A.shape[0]
    m = B.shape[0]

    rpAB, ciAB = L.indptr, L.indices
    nedges = len(ciAB)

    Si = []
    Sj = []

    wv = np.full(m, -1)
    ri1 = 0
    for i in range(n):
        for ri1 in range(rpAB[i], rpAB[i+1]):
            wv[ciAB[ri1]] = ri1

        for ip in A[i].nonzero()[1]:
            if i == ip:
                continue
            for ri2 in range(rpAB[ip], rpAB[ip+1]):
                jp = ciAB[ri2]
                for j in B[jp].nonzero()[1]:
                    if j == jp:
                        continue
                    if wv[j] >= 0:
                        Si.append(ri2)
                        Sj.append(wv[j])
        for ri1 in range(rpAB[i], rpAB[i+1]):
            wv[ciAB[ri1]] = -1

    return sps.csr_matrix(([1]*len(Si), (Sj, Si)), shape=(nedges, nedges), dtype=int)

def create_L(
    A: np.ndarray,
    B: np.ndarray,
    lalpha: float | None = 1,
    mind: float | None = None,
    weighted: bool = False,
) -> sps.csr_matrix:
    """Candidate match scores L[i, j] for aligning src node i to tar node j.

    Candidates are selected by degree proximity (window size lalpha). 
    If weighted, entries hold a degree-similarity score in (0, 1] else
    all entries at 1.0. Pairs scoring <= 0 are dropped unless
    mind is given as a floor value.
    """
    n = A.shape[0]
    m = B.shape[0]

    if lalpha is None:
        return sps.csr_matrix(np.ones((n, m)))

    a = A.sum(1)
    b = B.sum(1)

    a_p = list(enumerate(a))
    a_p.sort(key=lambda x: x[1])

    b_p = list(enumerate(b))
    b_p.sort(key=lambda x: x[1])

    ab_m = [0] * n
    s = 0
    e = floor(lalpha * log2(m))
    for ap in a_p:
        while(e < m and
              abs(b_p[e][1] - ap[1]) <= abs(b_p[s][1] - ap[1])
              ):
            e += 1
            s += 1
        ab_m[ap[0]] = [bp[0] for bp in b_p[s:e]]

    li = []
    lj = []
    lw = []
    for i, bj in enumerate(ab_m):
        for j in bj:
            d = 1 - abs(a[i] - b[j]) / max(a[i], b[j]) if weighted else 1.0
            if mind is None:
                if d > 0:
                    li.append(i)
                    lj.append(j)
                    lw.append(d)
            else:
                li.append(i)
                lj.append(j)
                lw.append(mind if d <= 0 else d)

    return sps.csr_matrix((lw, (li, lj)), shape=(n, m))