from __future__ import absolute_import
from builtins import range, zip
import autograd.numpy as np
import numpy as npo # original numpy
import autograd.scipy.signal

from numpy.lib.stride_tricks import as_strided
from future.utils import iteritems
import pickle
import time


def as_strided_seq(b, patch=5, stride=1):
    """
    Strides the input data sequentially through loops. For demonstration purposes only, use as_strided_dyn()
    for efficient implementation.
    :param b: array to be strided
    :param patch: patch is the length of one side of the patch. Must be smaller than smallest dimension of b
    :param stride: desired stride between patches
    :return: strided form of input data, prepared for convolution
    """
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
                            arr2 = np.array([b[k, :, i:i+patch, j:j+patch]])
                        else:
                            arr2 = npo.append(arr2, [b[k, :, i:i+patch, j:j+patch]], axis=0)
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


def as_strided_dyn(b, patch=5, stride=1):
    """
    Strides the input data as preprocessing for convolution.
    for efficient implementation
    :param b: array to be strided
    :param patch: patch is the length of one side of the patch. Must be smaller than smallest dimension of b
    :param stride: desired stride between patches
    :return: strided form of input data, prepared for convolution
    """
    dims = b.shape
    ex_ct = dims[0]
    if dims[3] != dims[4]:
        exit(-1)
    else:
        out = []
        for k in range(ex_ct):
            arr = []
            for i in range(0,dims[3]-patch+1,stride):
                arr2 = []
                for j in range(0,dims[4]-patch+1,stride):
                    if i+patch <= dims[3] and j+patch <= dims[4]:
                        arr2.append([b[k, :, :, i:i+patch, j:j+patch]])
                arr.append([arr2])
            out.append([arr])

    out = np.array(out)

    out = out[:, :, :, 0, :,  0, 0, :, :, :]
    out = np.moveaxis(out, 4, 2)

    return out


def convolve_seq(a, b):
    """
    Convolves two arrays sequentially through loops. For demonstration purposes only, use convolve_dyn()
    and convolve_tensor for efficient implementation.
    :param a: first array to be convoluted
    :param b: second array to be convoluted
    :return: convolved array
    """
    out = []
    a_dims = a.shape
    b = as_strided_seq(b, 5, 1)

    for ctr in range(a_dims[1]):
        if isinstance(a, np.ndarray):
            temp = a[0][ctr]
            tt = npo.flipud(temp)
            tt = npo.fliplr(tt)
            a[0][ctr] = tt
        else:
            a = a._value
            temp = a[0][ctr]
            tt = npo.flipud(temp)
            tt = npo.fliplr(tt)
            a[0][ctr] = tt

    if not isinstance(b[0][0][0][0][0][0], np.float):
        s = b.shape
        temp_b = np.empty(s)
        for i in range(s[0]):
            for j in range(s[1]):
                for k in range(s[2]):
                    for l in range(s[3]):
                        for m in range(s[4]):
                            for n in range(s[5]):
                                try:
                                    val = b[i][j][k][l][m][n]._value
                                    temp_b[i][j][k][l][m][n] = val
                                except:
                                    val = b[i][j][k][l][m][n]
                                    temp_b[i][j][k][l][m][n] = val


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
                    #filter = filter[0]
                    patch = b[ctr, j, k, :, :, :]
                    try:
                        temp = npo.einsum('ijk,ijk->', filter, patch)
                    except:
                        #print("boo")
                        patch = temp_b[ctr, j, k, :, :, :]
                        temp = npo.einsum('ijk,ijk->', filter, patch)
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


def convolve_seq_iter(a, b):
    """
    Convolves two arrays. For tensorized implementation, see convolve_tensor().
    :param a: first array to be convoluted
    :param b: second array to be convoluted
    :return: convolved array
    """
    a_dims = a.shape
    b = as_strided_seq(b, 5, 1)  # arbitrary patch & stride for now
    di = DataIterator(b)

    for ctr in range(a_dims[1]):
        if isinstance(a, np.ndarray):
            temp = a[0][ctr]
            tt = npo.flipud(temp)
            tt = npo.fliplr(tt)
            a[0][ctr] = tt
        else:
            a = a._value
            temp = a[0][ctr]
            tt = npo.flipud(temp)
            tt = npo.fliplr(tt)
            a[0][ctr] = tt

    out = []
    for i in range(a_dims[1]):
        filters = []
        filter = a[:, i, :, :]
        di.reset()
        while di.has_next():
            patch = di.next()
            try:
                temp = npo.einsum('ijk,ijk->', filter, patch)
            except:
                print("Convolution unsuccessful, exit op")
            filters.append(temp)
        if len(out) == 0:
            out = npo.array([filters])
        else:
            out = npo.vstack((out, [filters]))

    out = out.reshape((a_dims[1], b.shape[0], b.shape[1], b.shape[2]))
    out = np.swapaxes(out, 0, 1)
    return out


def convolve_tensor(a, b):
    """
    Convolves two tensorized arrays, most efficient convolution method.
    :param a: first array to be convoluted
    :param b: second array to be convoluted
    :return: convolved array
    """
    b = as_strided_seq(b, 5, 1)
    b = np.moveaxis(b, [0, 1, 2, 3, 4, 5], [0, 3, 4, 5, 1, 2])
    b = np.moveaxis(b, 5, 1)
    try:
        out = npo.einsum(a, [12, 1, 10, 11], b, [4, 12, 10, 11, 8, 9])
        out = np.swapaxes(out, 0, 1)
    except:
        a = a._value
        out = npo.einsum(a, [12, 1, 10, 11], b, [4, 12, 10, 11, 8, 9])
        out = np.swapaxes(out, 0, 1)
    return out


class DataIterator:     # Alternative impl: do not copy the array but access elements inplace.
    """
    Iterator to access input sequentially.
    """
    def __init__(self, b):
        self.ix = 0
        self.arr = []
        dims = b.shape
        self.length = dims[0] * dims[1] * dims[2]
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    self.arr.append(b[i, j, k, :, :, :])

    def has_next(self):
        return False if self.ix >= self.length else True

    def next(self):
        try:
            item = self.arr[self.ix]
        except:
            print("Error: next failed")
        self.ix += 1
        return item

    def reset(self):
        self.ix = 0


def mnist_example():
    """
    Runs a toy convolution example.
    :return: None
    """
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