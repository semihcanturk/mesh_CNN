from __future__ import absolute_import
from builtins import range, zip
import autograd.numpy as np
import numpy as npo # original numpy
import autograd.scipy.signal

from numpy.lib.stride_tricks import as_strided
from future.utils import iteritems
import mnist
from conv import hips_convnet
import pickle

def as_strided_seq(b, patch, stride):
    # b is array to be strided
    # patch is the length of one side of the patch. Must be smaller than smallest dimension of b
    # stride is how much of a stride we want, we may wanna default it to 1
    # TODO: How to deal with padding?
    dims = b.shape
    ex_ct = dims[0]
    if dims[2] != dims[3]:
        exit(-1)
    else:
        out = []
        for k in range(ex_ct):
            arr = []
            for i in range(0,dims[2]-patch+1,stride):
                arr2 = []
                for j in range(0,dims[3]-patch+1,stride):
                    if i+patch <= dims[2] and j+patch <= dims[3]:
                        if len(arr2) == 0:
                            arr2 = np.array(b[k, :, i:i+patch, j:j+patch])
                        else:
                            arr2 = npo.append(arr2, b[k, :, i:i+patch, j:j+patch], axis=0)
                    #potential ELSE here
                if len(arr2) == dims[2]-patch+1:
                    if len(arr) == 0:
                        arr = np.array([arr2])
                        arr2 = []
                    else:
                        arr = npo.vstack((arr, [arr2]))
                        arr2 = []
            if len(out) == 0:
                out = np.array([arr])
                arr = []
            else:
                out = npo.vstack((out, [arr]))
                arr = []
    return out


def convolve_seq(a, b):
    out = []
    a_dims = a.shape
    b = as_strided_seq(b, 5, 1)  # arbitrary patch & stride for now

    for ctr in range(a_dims[1]):
        temp = a[0][ctr]
        tt = npo.flipud(temp)
        tt = npo.fliplr(tt)
        a[0][ctr] = tt

    b_dims = b.shape
    ex_ct = b_dims[0]
    for ctr in range(ex_ct):
        filters = []
        for i in range(a_dims[1]):
            arr = []
            for j in range(b_dims[1]):
                row = []
                for k in range(b_dims[2]):
                    filter = a[:, i, :, :]
                    filter = filter[0]
                    patch = b[ctr, j, k, :, :]
                    temp = npo.einsum('ij,ij->', filter, patch)
                    row.append(temp)
                if len(arr) == 0:
                    arr = npo.array([row])
                    row = []
                else:
                    arr = npo.vstack((arr, [row]))
                    row = []
            if len(filters) == 0:
                filters = npo.array([arr])
                arr = []
            else:
                filters = npo.vstack((filters, [arr]))
                arr = []
        if len(out) == 0:
            out = npo.array([filters])
            filters = []
        else:
            out = npo.vstack((out, [filters]))
            filters = []
    return out


def mnist_example():
    B_file = open('../data/b.pickle', 'rb')
    B = pickle.load(B_file)  # variables come out in the order you put them in
    B_file.close()

    A_file = open('../data/a.pickle', 'rb')
    A = pickle.load(A_file)  # variables come out in the order you put them in
    A_file.close()

    B = B[:100]
    conv = convolve_seq(A, B)
    print(conv.shape)
    print(conv)


if __name__ == '__main__':
    mnist_example()