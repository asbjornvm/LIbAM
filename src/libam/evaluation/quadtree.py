import numpy as np
import torch


def quad_propagate(target, source, i, dim=0):
    _i = i << 1
    if dim == 0:
        xx = torch.amax(source[_i: _i + 2], dim=dim)
    else:
        xx = torch.amax(source[:, _i: _i + 2], dim=dim)
    if dim == 0:
        target[i] = torch.amax(xx.view(-1, 2), dim=1)
    else:
        target[:, i] = torch.amax(xx.view(-1, 2), dim=1)


def dequad(array):
    M = torch.empty((array.shape[0]//2, array.shape[1]//2))

    for i in range(M.shape[1]):

        quad_propagate(M, array, i)

    return M


def build_quadtree(matrix):
    quadtree = [matrix]
    while quadtree[-1].shape != (1, 1):
        node = dequad(quadtree[-1])
        quadtree.append(node)

    return quadtree


def lookup(quadtree):
    x = y = 0
    for node in quadtree[::-1]:
        x *= 2
        y *= 2
        _x, _y = np.unravel_index(
            torch.argmax(node[x:x+2, y:y+2]),
            shape=(2, 2)
        )
        x += _x
        y += _y
    return x, y


def reconstruct_pytorch(quadtree, arg):
    x, y = arg
    quadtree[0][x] = float('-inf')
    quadtree[0][:, y] = float('-inf')
    for i in range(1, len(quadtree)):
        current_node = quadtree[i]
        prev_node = quadtree[i-1]

        x >>= 1

        quad_propagate(current_node, prev_node, x)

        y >>= 1

        quad_propagate(current_node, prev_node, y, dim=1)

def superfast_binbin_torch(M):
    M = torch.from_numpy(M)
    n = M.shape[0]

    quadtree = build_quadtree(M)

    ma = np.zeros(n, int)
    mb = np.zeros(n, int)

    for _ in range(n):

        argmax = lookup(quadtree)

        reconstruct_pytorch(quadtree, argmax)

        x, y = argmax
        ma[x] = x
        mb[x] = y

    return ma, mb