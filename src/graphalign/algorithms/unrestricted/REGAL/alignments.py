import numpy as np
import scipy.io as sio
import sklearn.metrics.pairwise
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import KDTree
import scipy.sparse as sp
from scipy.spatial.distance import cosine


def get_embedding_similarities(embed, embed2=None, sim_measure="euclidean", num_top=None):
    n_nodes, dim = embed.shape
    if embed2 is None:
        embed2 = embed

    # if num_top is not None:  # KD tree with only top similarities computed
    # similarity_matrix = None
    similarity_matrix = kd_align(
        embed, embed2, distance_metric=sim_measure, num_top=num_top)
    return similarity_matrix, sklearn.metrics.pairwise.euclidean_distances(
        embed, embed2)
    # return None, -similarity_matrix

# Split embeddings in half (TODO generalize to different numbers and sizes of networks)


def get_embeddings(combined_embed):
    # right now assume graphs are same size
    n_nodes = combined_embed.shape[0] / 2
    dim = combined_embed.shape[1]
    n_nodes = int(n_nodes)
    embed1 = combined_embed[:n_nodes]
    embed2 = combined_embed[n_nodes:]
    return embed1, embed2


def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=50):
    kd_tree = KDTree(emb2, metric=distance_metric)

    row = np.array([])
    col = np.array([])
    data = np.array([])
    # change later
    dist, ind = kd_tree.query(emb1, k=num_top)
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top)*i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix(
        (data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix.tocsr()