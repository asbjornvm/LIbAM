# -*- coding: utf-8 -*-
"""
Rewrite ot.bregman.sinkhorn in Python Optimal Transport
(https://pythonot.github.io/_modules/ot/bregman.html#sinkhorn).
Bregman projections for regularized OT (Sinkhorn distance).
"""
import torch

M_EPS = 1e-16
TOL = 1e-5


def SinkhornKnopp(C, a, b, sinkhorn_iters=100, tol=TOL, evalfreq=10):
    device = a.device
    na, nb = C.shape
    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Size of a(b) does't match C"
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    u = torch.ones(na, dtype=a.dtype).to(device) / na
    v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    K = 1.0*C

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1

    # allocate memory beforehand
    KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    Kv = torch.empty(u.shape, dtype=u.dtype).to(device)

    while (err > tol and it <= sinkhorn_iters):
        torch.matmul(u, K, out=KTu)
        v = torch.div(b, KTu)
        torch.matmul(K, v, out=Kv)
        u = torch.div(a, Kv)

        if it % evalfreq == 0:
            if torch.any(KTu == 0) or torch.any(torch.isnan(u)) or \
                torch.any(torch.isnan(v)) or torch.any(torch.isinf(u)) or \
                    torch.any(torch.isinf(v)):
                raise RuntimeError("Sinkhorn-Knopp failed to converge.")
            if len(a) != len(b):
                b_hat = torch.matmul(u, K) * v
                a_hat = torch.matmul(v, K.T) * u
                err_a = (a - a_hat).pow(2).sum().item()
                err_b = (b - b_hat).pow(2).sum().item()
                err = (err_a + err_b)/2
            else:
                b_hat = torch.matmul(u, K) * v
                err = (b - b_hat).pow(2).sum().item()
        it += 1

    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    return P
