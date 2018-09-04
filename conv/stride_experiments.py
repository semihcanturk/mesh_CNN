from __future__ import absolute_import
from builtins import range, zip
from functools import partial
import autograd.numpy as np
import numpy as npo # original numpy
from autograd.extend import primitive, defvjp

from numpy.lib.stride_tricks import as_strided
from future.utils import iteritems

def pad_to_full(A, B, axes):
    A_pad = [(0, 0)] * A.ndim
    for ax_A, ax_B in zip(*axes):
        A_pad[ax_A] = (B.shape[ax_B] - 1,) * 2
    return npo.pad(A, A_pad, mode='constant')

def convolve(A, B, axes=None, dot_axes=[(),()], mode='full'):
    assert mode in ['valid', 'full'], "Mode {0} not yet implemented".format(mode)
    if axes is None:
        axes = [list(range(A.ndim)), list(range(A.ndim))]
    wrong_order = any([B.shape[ax_B] < A.shape[ax_A] for ax_A, ax_B in zip(*axes)])
    if wrong_order:
        if mode=='valid' and not all([B.shape[ax_B] <= A.shape[ax_A] for ax_A, ax_B in zip(*axes)]):
                raise Exception("One array must be larger than the other along all convolved dimensions")
        elif mode != 'full' or B.size <= A.size: # Tie breaker
            i1 =      B.ndim - len(dot_axes[1]) - len(axes[1]) # B ignore
            i2 = i1 + A.ndim - len(dot_axes[0]) - len(axes[0]) # A ignore
            i3 = i2 + len(axes[0])
            ignore_B = list(range(i1))
            ignore_A = list(range(i1, i2))
            conv     = list(range(i2, i3))
            return convolve(B, A, axes=axes[::-1], dot_axes=dot_axes[::-1], mode=mode).transpose(ignore_A + ignore_B + conv)

    if mode == 'full':
        B = pad_to_full(B, A, axes[::-1])
    B_view_shape = list(B.shape)
    B_view_strides = list(B.strides)
    flipped_idxs = [slice(None)] * A.ndim
    for ax_A, ax_B in zip(*axes):
        B_view_shape.append(abs(B.shape[ax_B] - A.shape[ax_A]) + 1)
        B_view_strides.append(B.strides[ax_B])
        B_view_shape[ax_B] = A.shape[ax_A]
        flipped_idxs[ax_A] = slice(None, None, -1)

    B_view = as_strided(B, B_view_shape, B_view_strides)
    return B_view




#Examples follow

#A = np.ndarray(shape=(1,6,5,5), dtype=np.float64)
#A.fill(2)
#B = np.ndarray(shape=(1000,1,28,28), dtype=np.float64)
#B.fill(5)

A = np.random.randint(0,99, size=(1,6,5,5), dtype=np.int64)
B = np.random.randint(0,99, size=(1000,1,28,28), dtype=np.int64)

def ex1():
    conv = convolve(A, B, axes=([2, 3], [2, 3]), dot_axes = ([1], [0]), mode='valid')
    return conv

def ex2():
    a = np.arange(60.).reshape(3,4,5)
    b = np.arange(24.).reshape(4,3,2)
    res = npo.einsum(a, [0,1,2], b, [1,0,3])
    res2 = npo.einsum(a, [0,1,2], b, [1,0,3], [2,3])
    print(res == res2)

def ex3():
    B_ex = convolve(A, B, axes=([2, 3], [2, 3]), dot_axes=([1], [0]), mode='valid')
    res = npo.einsum(A,[12,1,10,11],B_ex,[4,12,10,11,8,9])
    res2 = npo.einsum(A,[12,1,10,11],B_ex,[4,12,10,11,8,9],[1,4,8,9])
    res3 = npo.einsum(A, [6, 2, 5, 22], B_ex, [7, 6, 5, 22, 4, 18], [2, 7, 4, 18])
    return res, res2, res3

p, p2, p3 = ex3()
print(p)
print(p.all() == p2.all() and p2.all() == p3.all())