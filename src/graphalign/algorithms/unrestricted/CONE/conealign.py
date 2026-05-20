import numpy as np
import sklearn.metrics.pairwise
import scipy.sparse as sps
import time
from dataclasses import dataclass
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import KDTree

from . import unsup_align, embedding
from graphalign.algorithms.algorithm import Algorithm
from graphalign import GraphPair


# original code from https://github.com/GemsLab/CONE-Align


def align_embeddings(
    embed1: np.ndarray,
    embed2: np.ndarray,
    cone_args: dict,
    adj1=None,
    adj2=None,
    struc_embed=None,
    struc_embed2=None,
) -> tuple[csr_matrix, np.ndarray]:
    corr = None
    if struc_embed is not None and struc_embed2 is not None:
        if cone_args["embsim"] == "cosine":
            corr = sklearn.metrics.pairwise.cosine_similarity(embed1, embed2)
        else:
            corr = sklearn.metrics.pairwise.euclidean_distances(embed1, embed2)
            corr = np.exp(-corr)

        matches = np.zeros(corr.shape)
        matches[np.arange(corr.shape[0]), np.argmax(corr, axis=1)] = 1
        corr = matches

    if adj1 is not None and adj2 is not None:
        if not sps.issparse(adj1):
            adj1 = sps.csr_matrix(adj1)
        if not sps.issparse(adj2):
            adj2 = sps.csr_matrix(adj2)
        init_sim, _ = unsup_align.convex_init_sparse(
            embed1, embed2, K_X=adj1, K_Y=adj2,
            apply_sqrt=False, niter=cone_args["niter_init"], reg=cone_args["reg_init"], P=corr,
        )
    else:
        init_sim, _ = unsup_align.convex_init(
            embed1, embed2,
            apply_sqrt=False, niter=cone_args["niter_init"], reg=cone_args["reg_init"],
        )

    dim_align_matrix, _ = unsup_align.align(
        embed1, embed2, init_sim,
        lr=cone_args["lr"], bsz=cone_args["bsz"], nepoch=cone_args["nepoch"],
        niter=cone_args["niter_align"], reg=cone_args["reg_align"],
    )

    aligned_embed1 = embed1.dot(dim_align_matrix)
    alignment_matrix = kd_align(
        aligned_embed1, embed2,
        distance_metric=cone_args["embsim"], num_top=cone_args["numtop"],
    )
    cost_matrix = sklearn.metrics.pairwise.euclidean_distances(aligned_embed1, embed2)
    return alignment_matrix, cost_matrix


def kd_align(
    emb1: np.ndarray,
    emb2: np.ndarray,
    distance_metric: str = "euclidean",
    num_top: int = 10,
) -> csr_matrix:
    kd_tree = KDTree(emb2, metric=distance_metric)
    dist, ind = kd_tree.query(emb1, k=num_top)

    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()

    sparse_align_matrix = coo_matrix(
        (data, (row, col)), shape=(emb1.shape[0], emb2.shape[0])
    )
    return sparse_align_matrix.tocsr()


@dataclass
class Cone(Algorithm):
    pair: GraphPair
    dim: int
    window: int
    negative: float
    niter_init: int
    reg_init: float
    lr: float
    bsz: int
    nepoch: int
    niter_align: int
    reg_align: float
    embsim: str
    numtop: int

    @property
    def name(self) -> str:
        return "CONE"

    def evaluate(self) -> np.ndarray:
        src: np.ndarray = self.pair.src_adjacency
        tar: np.ndarray = self.pair.tar_adjacency

        dim = min(self.dim, src.shape[0] - 1)

        start = time.time()
        emb_matrix_a = embedding.netmf(
            src, dim=dim, window=self.window, b=self.negative, normalize=True,
        )
        emb_matrix_b = embedding.netmf(
            tar, dim=dim, window=self.window, b=self.negative, normalize=True,
        )

        cone_args = {
            "niter_init": self.niter_init, "reg_init": self.reg_init,
            "lr": self.lr, "bsz": self.bsz, "nepoch": self.nepoch,
            "niter_align": self.niter_align, "reg_align": self.reg_align,
            "embsim": self.embsim, "numtop": self.numtop,
        }
        alignment_matrix, _ = align_embeddings(
            emb_matrix_a, emb_matrix_b, cone_args,
            adj1=csr_matrix(src), adj2=csr_matrix(tar),
        )

        total_time = time.time() - start
        print(f"time for CONE-align (in seconds): {total_time:.4f}")

        return alignment_matrix.toarray()