import scipy.optimize as opt
import numpy as np
from . import grasp as fu
import sys
import matplotlib.pyplot as plt

import torch
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.optimizers import SteepestDescent

D1 = None
D2 = None
V1 = None
V2 = None
Cor1 = None
Cor2 = None
k_ = None


def cost(X):
    mu = 0.132

    global D2
    global V1
    global V2
    global Cor1
    global Cor2
    global k_
    coup = (torch.linalg.norm(Cor1.T@V1[:, 0:k_]-Cor2.T@V2[:, 0:k_]@X, 'fro'))**2

    res = (X.T @ torch.diag(D2[0:k_]) @ X) ** 2
    diag_res = torch.diagonal(res, offset=0, dim1=-1, dim2=-2)

    diag_res = torch.sum(diag_res)

    sumres = torch.sum(res)

    val = sumres - diag_res
    res = val+mu*coup
    return res


def optimize_AB(Cor11, Cor21, n, V11, V21, D11, D21, k):

    global D2
    global V1
    global V2
    global Cor1
    global Cor2
    global k_

    k_ = k
    x0 = init_x0(Cor11, Cor21, n, V11, V21, D11, D21, k)

    D2 = torch.tensor(D21, dtype=torch.float64)
    V1 = torch.tensor(V11, dtype=torch.float64)
    V2 = torch.tensor(V21, dtype=torch.float64)
    Cor1 = torch.tensor(Cor11, dtype=torch.float64)
    Cor2 = torch.tensor(Cor21, dtype=torch.float64)

    manifold = Stiefel(k, k)
    decorated_cost = pymanopt.function.pytorch(manifold)(cost)
    problem = Problem(manifold, decorated_cost)

    # (3) Instantiate a Pymanopt solver

    solver = pymanopt.optimizers.TrustRegions(verbosity=0)

    # let Pymanopt do the rest
    B = solver.run(problem, initial_point=x0).point

    return B


def init_x0(Cor1, Cor2, n, V1, V2, D1, D2, k):

    B = np.identity(k)

    for i in range(0, k):
        thing1 = np.linalg.norm(Cor1.T@V1[:, i]-Cor2.T@V2[:, i])
        thing2 = np.linalg.norm(Cor1.T@V1[:, i]+Cor2.T@V2[:, i])
        if(thing1 > thing2):
            B[i, i] = -1

    return B
