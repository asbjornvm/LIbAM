from dataclasses import dataclass
import scipy.sparse as sps
import numpy as np

from graphalign import GraphPair
from graphalign.algorithms import bipartitewrapper as bmw
from graphalign.algorithms.algorithm import Algorithm


#original code https://www.cs.purdue.edu/homes/dgleich/codes/netalign/

def othermaxplus(dim, li, lj, lw, m, n):

    if dim == 1:
        i1 = lj
        i2 = li
        N = n
    else:
        i1 = li
        i2 = lj
        N = m

    dimmax1 = np.zeros(N)
    dimmax2 = np.zeros(N)
    dimmaxind = np.zeros(N)
    nedges = len(li)

    for i in range(nedges):
        if lw[i] > dimmax2[i1[i]]:
            if lw[i] > dimmax1[i1[i]]:
                dimmax2[i1[i]] = dimmax1[i1[i]]
                dimmax1[i1[i]] = lw[i]
                dimmaxind[i1[i]] = i2[i]
            else:
                dimmax2[i1[i]] = lw[i]

    omp = np.zeros(len(lw))
    for i in range(nedges):
        if i2[i] == dimmaxind[i1[i]]:
            omp[i] = dimmax2[i1[i]]
        else:
            omp[i] = dimmax1[i1[i]]

    return omp


def othersum(si, sj, s, m, n):
    rowsum = accumarray(si, s, m)
    return rowsum[si] - s


def accumarray(xij, xw, n):
    sums = np.zeros(n)

    for i in range(len(xij)):
        sums[xij[i]] += xw[i]

    return sums

@dataclass
class NetAlign(Algorithm):
    pair: GraphPair
    a: int
    b: int
    gamma: float
    dtype:int
    max_iter:int
    verbose: bool

    @property
    def name(self) -> str:
        return "NetAlign"

    def evaluate(self) -> np.ndarray:
        S = self.pair.S
        li = self.pair.li
        lj = self.pair.lj
        w = self.pair.w

        S = sps.csr_matrix(S)

        n_edges = len(li)
        n_squares = S.count_nonzero() // 2

        # compute a vector that allows us to transpose data between squares.
        sui, suj, _ = sps.find(sps.triu(S, 1))
        SI = sps.csr_matrix(
            (list(range(1, len(sui)+1)), (sui, suj)), shape=S.shape
        )

        SI = SI + SI.transpose()
        si, sj, sind = sps.find(SI)
        sind = [x-1 for x in sind]
        SP = sps.csr_matrix(
            ([1]*len(si), (si, sind)), shape=(S.shape[0], n_squares)
        )
        sij, sijrs, _ = sps.find(SP)
        sind = list(range(SP.count_nonzero()))
        spair = sind[:]
        spair[::2] = sind[1::2]
        spair[1::2] = sind[::2]

        # Initialize the messages
        ma = np.zeros(n_edges, int)
        mb = np.zeros(n_edges, int)
        ms = np.zeros(S.count_nonzero())
        sums = np.zeros(n_edges)

        damping = self.gamma
        curdamp = 1
        alpha = self.a
        beta = self.b

        fbest = 0
        fbestiter = 0
        if self.verbose:
            print('{:4s}   {:>4s}   {:>7s} {:>7s} {:>7s} {:>7s}   {:>7s} {:>7s} {:>7s} {:>7s}'.format(
                'best', 'iter', 'obj_ma', 'wght_ma', 'card_ma', 'over_ma',
                'obj_mb', 'wght_mb', 'card_mb', 'over_mb'))

        setup, m, n = bmw.bipartite_setup(li, lj, w)

        for it in range(1, self.max_iter + 1):
            prevma = ma
            prevmb = mb
            prevms = ms
            prevsums = sums
            curdamp = damping * curdamp

            omaxb = np.array(
                [max(0, x) for x in othermaxplus(2, li, lj, mb, m, n)],
            )
            omaxa = np.array(
                [max(0, x)for x in othermaxplus(1, li, lj, ma, m, n)],
            )

            msflip = ms[spair]
            mymsflip = msflip+beta
            mymsflip = [min(beta, x) for x in mymsflip]
            mymsflip = [max(0, x) for x in mymsflip]

            sums = accumarray(sij, mymsflip, n_edges)

            ma = alpha*w - omaxb + sums
            mb = alpha*w - omaxa + sums

            ms = alpha*w[sij]-(omaxb[sij] + omaxa[sij])
            ms += othersum(sij, sijrs, mymsflip, n_edges, n_squares)

            if self.dtype == 1:
                ma = curdamp*(ma) + (1-curdamp)*(prevma)
                mb = curdamp*(mb) + (1-curdamp)*(prevmb)
                ms = curdamp*(ms) + (1-curdamp)*(prevms)
            elif self.dtype == 2:
                ma = ma + (1-curdamp)*(prevma+prevmb-alpha*w+prevsums)
                mb = mb + (1-curdamp)*(prevmb+prevma-alpha*w+prevsums)
                ms = ms + (1-curdamp)*(prevms+prevms[spair]-beta)
            elif self.dtype == 3:
                ma = curdamp*ma + (1-curdamp)*(prevma+prevmb-alpha*w+prevsums)
                mb = curdamp*mb + (1-curdamp)*(prevmb+prevma-alpha*w+prevsums)
                ms = curdamp*ms + (1-curdamp)*(prevms+prevms[spair]-beta)

            hista = bmw.round_messages(
                ma, S, w, alpha, beta, setup, m, n)[:-2]

            histb = bmw.round_messages(
                mb, S, w, alpha, beta, setup, m, n)[:-2]

            if hista[0] > fbest:
                fbestiter = it
                mbest = ma
                fbest = hista[0]
            if histb[0] > fbest:
                fbestiter = -it
                mbest = mb
                fbest = histb[0]

            if self.verbose:
                if fbestiter == it:
                    bestchar = '*a'
                elif fbestiter == -it:
                    bestchar = '*b'
                else:
                    bestchar = ''
                print('{:4s}   {:4d}   {:7.2f} {:7.2f} {:7d} {:7d}   {:7.2f} {:7.2f} {:7d} {:7d}'.format(
                    bestchar, it, *hista, *histb))

        return sps.csr_matrix((mbest, (li, lj))).toarray()
