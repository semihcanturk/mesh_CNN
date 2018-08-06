from __future__ import absolute_import
from builtins import range, zip
from functools import partial
import autograd.numpy as np
import numpy as npo # original numpy
import autograd.scipy.signal
from autograd.extend import primitive, defvjp

from numpy.lib.stride_tricks import as_strided
from future.utils import iteritems
import mnist
import hips_convnet
import pickle

convolve = autograd.scipy.signal.convolve
A_default = hips_convnet.A_def
B_default = hips_convnet.A_def


def convolve_semih(a, b):
    out = []
    a_dims = a.shape
    b = as_strided_semih(b, 5, 1)  # arbitrary patch & stride for now

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
        #filters = filters.reshape(6, 24, 24) #TODO: Is this legit?
        if len(out) == 0:
            out = npo.array([filters])
            filters = []
        else:
            out = npo.vstack((out, [filters]))
            filters = []
    return out


#TODO
def einsum_semih(a, b, a_axnums, b_axnums):
    return None


#TODO
def as_strided_semih(b, patch, stride):
    #b is array to be strided
    #patch is the length of one side of the patch. Must be smaller than smallest dimension of b
    #stride is how much of a stride we want, we may wanna default it to 1
    #TODO: How to deal with padding?
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




def einsum_tensordot(A, B, axes, reverse=False):
    # Does tensor dot product using einsum, which shouldn't require a copy.
    A_axnums = list(range(A.ndim))
    B_axnums = list(range(A.ndim, A.ndim + B.ndim))
    sum_axnum = A.ndim + B.ndim
    for i_sum, (i_A, i_B) in enumerate(zip(*axes)):
        A_axnums[i_A] = sum_axnum + i_sum
        B_axnums[i_B] = sum_axnum + i_sum
    return einsum_semih(A, A_axnums, B, B_axnums)


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
    #B_view_2 = as_strided_semih(B, B_view_shape, B_view_strides)
    #stride = 1
    #radius = B_view_shape[4]
    #c = (radius/2) - 1
    #center = [c, c]
    #B_view = as_strided_v2(B, radius, center, stride)
    A_view = A[flipped_idxs]
    all_axes = [list(axes[i]) + list(dot_axes[i]) for i in [0, 1]]
    return einsum_tensordot(A_view, B_view, all_axes)


#print("Loading training data...")
#add_color_channel = lambda x : x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
#one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)

#train_images = mnist.train_images()
#train_labels = mnist.train_labels()

#test_images = mnist.test_images()
#test_labels = mnist.test_labels()

#train_images = add_color_channel(train_images) / 255.0
#test_images  = add_color_channel(test_images)  / 255.0
#train_labels = one_hot(train_labels, 10)
#test_labels = one_hot(test_labels, 10)
#N_data = train_images.shape[0]




if __name__ == '__main__':
    print("as_strided_semih test")
    A = np.random.randint(0, 99, size=(1, 6, 5, 5), dtype=np.int64)
    B = np.random.randint(0, 99, size=(10, 1, 28, 28), dtype=np.int64)
    B_view_semih = as_strided_semih(B, 5, 1)

    #B_file = open('b.pickle', 'rb')
    #B_2 = pickle.load(B_file)  # variables come out in the order you put them in
    #B_file.close()

    #A_file = open('a.pickle', 'rb')
    #A_2 = pickle.load(A_file)  # variables come out in the order you put them in
    #A_file.close()

    conv = convolve(B, A, axes=([2, 3], [2, 3]), dot_axes=([1], [0]), mode='valid')

    #B_2 = B_2[:100]
    #B_view_semih_2 = as_strided_semih(B_2, 5, 1)
    conv_semih = convolve_semih(A, B)
    #conv_by_einsum = npo.einsum(A_2, [12, 1, 10, 11], B_view_semih_2, [4, 8, 9, 10, 11])
    print(conv_semih.shape)
    print(conv_semih)