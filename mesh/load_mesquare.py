import numpy as np
from conv import convolution_impl, mnist
import pickle
import pysal
import time
from numpy import genfromtxt
import sys
#from scipy import sparse
import sparse
import os
import h5py
import hickle as hkl


def load():
    """
    creates a square gridmesh equal to the dimensions of the mnist dataset, and embeds the mnist examples on the mesh
    structure, allowing the mesh cnn to discriminate the numbers from the mesh instead of the dataset.
    :return: train/test and mesh data
    """

    # image dimensions
    M = 28
    N = 28

    # Load and process MNIST data
    print("Loading training data...")

    mnist.init()    # run this only the first time the code is run
    train_images, train_labels, test_images, test_labels = mnist.load()

    train_images = train_images.reshape((train_images.shape[0], 1, 28, 28)) / 255.0
    test_images = test_images.reshape((test_images.shape[0], 1, 28, 28)) / 255.0

    train_images = train_images[:10000, :, :]
    train_labels = train_labels[:10000]
    test_images = test_images[:500, :, :]
    test_labels = test_labels[:500]

    try:
        train_batch = pickle.load(open("train_data.pickle", "rb"))
        test_batch = pickle.load(open("test_data.pickle", "rb"))
    except (OSError, IOError) as e:
        train_batch = convolution_impl.as_strided_seq(train_images, 5, 1)
        test_batch = convolution_impl.as_strided_seq(test_images, 5, 1)
        pickle.dump(train_batch, open("train_data.pickle", "wb"))
        pickle.dump(test_batch, open("test_data.pickle", "wb"))

    train_batch = train_batch.reshape(train_batch.shape[0], train_batch.shape[1] * train_batch.shape[2],
                                      train_batch.shape[4] * train_batch.shape[5])
    test_batch = test_batch.reshape(test_batch.shape[0], test_batch.shape[1] * test_batch.shape[2],
                                      test_batch.shape[4] * test_batch.shape[5])

    X = np.random.rand(M, N)

    adj_mtx, Y = embed_mesh(X)

    coords = []
    for i in range(M):
        for j in range(N):
            coords.append([i, j, 0])

    coords = np.array(coords)
    return train_batch, train_labels, test_batch, test_labels, adj_mtx, coords


def load_dynamic(n_ex=None):
    """
    creates a square gridmesh equal to the dimensions of the mnist dataset, and embeds the mnist examples on the mesh
    structure, allowing the mesh cnn to discriminate the numbers from the mesh instead of the dataset. Then, sparsifies
    the examples by adding a time dimension, with only a section of the full image shown in each time step. this results
    in a dynamic recognition task rather than a static one, which was the case in load().
    :return: train/test and mesh data
    """

    # image dimensions
    M = 28
    N = 28

    H = 14

    # Load and process MNIST data
    print("Loading training data...")

    mnist.init()  # run this only the first time the code is run
    train_images, train_labels, test_images, test_labels = mnist.load()

    train_images = train_images.reshape((train_images.shape[0], 1, 28, 28)) / 255.0
    test_images = test_images.reshape((test_images.shape[0], 1, 28, 28)) / 255.0

    if n_ex is not None:
        train_images = train_images[:n_ex, :, :]
        train_labels = train_labels[:n_ex]
        test_images = test_images[:int(n_ex/10), :, :]
        test_labels = test_labels[:int(n_ex/10)]

    stime = time.time()

    try:
        with h5py.File('dyn_sparse.h5', 'r') as hf:
            train_batch = hf['dyn_train'][:]
            test_batch = hf['dyn_test'][:]

    except (OSError, IOError) as e:

        dyn_train = []
        for img in train_images:
            new_img = []
            for i in range(0, H):
                mask = np.zeros((1, img.shape[1], img.shape[2]))
                mask[0, 2 * i] = 1
                mask[0, 2 * i + 1] = 1
                temp = np.multiply(mask, img)
                new_img.append([temp])
            dyn_train.append([new_img])

        dyn_test = []
        for img in test_images:
            new_img = []
            for i in range(0, H):
                mask = np.zeros((1, img.shape[1], img.shape[2]))
                mask[0, 2 * i] = 1
                mask[0, 2 * i + 1] = 1
                temp = np.multiply(mask, img)
                new_img.append([temp])
            dyn_test.append([new_img])

        dyn_train = np.array(dyn_train)
        dyn_test = np.array(dyn_test)

        dyn_train = dyn_train[:, :, :, 0, 0, : :]
        dyn_test = dyn_test[:, :, :, 0, 0, : :]

        train_batch = convolution_impl.as_strided_dyn(dyn_train, 5, 1)
        test_batch = convolution_impl.as_strided_dyn(dyn_test, 5, 1)

        train_sparse = sparse.COO.from_numpy(train_batch)
        test_sparse = sparse.COO.from_numpy(test_batch)

        with h5py.File('dyn_sparse.h5', 'w') as hf:
            hf.create_dataset('dyn_train', data=train_sparse, compression="gzip", compression_opts=9)
            hf.create_dataset('dyn_test', data=test_sparse, compression="gzip", compression_opts=9)

    etime = time.time()

    print(etime-stime)

    train_batch = train_batch.reshape(train_batch.shape[0], train_batch.shape[2],
                                      train_batch.shape[3] * train_batch.shape[4],
                                      train_batch.shape[5] * train_batch.shape[6])
    test_batch = test_batch.reshape(test_batch.shape[0], test_batch.shape[2],
                                    test_batch.shape[3] * test_batch.shape[4],
                                    test_batch.shape[5] * test_batch.shape[6])

    X = np.random.rand(M, N)

    adj_mtx, Y = embed_mesh(X)

    coords = []
    for i in range(M):
        for j in range(N):
            coords.append([i, j, 0])

    coords = np.array(coords)
    return train_batch, train_labels, test_batch, test_labels, adj_mtx, coords


def embed_mesh(X):

    (M, N) = X.shape

    # adjacency matrix
    A = np.zeros((M * N, M * N))

    # vector of node values
    Y = np.zeros((M * N, 1))

    for j in range(N):
        for i in range(M):

            # node id
            k = (j - 1) * M + i

            # edge north
            if i > 1:
                A[k, k - 1] = 1
            # edge south
            if i < M:
                A[k, k + 1] = 1
            # edge west
            if j > 1:
                A[k, k - M] = 1
            # edge east
            if j < N:
                A[k, k + M] = 1

    return A, Y


def create_mtx(x):
    """
    creates adjacency matrix
    :param x: size of matrix
    :return: adjacency matrix
    """
    X = np.random.rand(x, x)

    # optional way to create adjacency matrix
    # adj_mtx, _ = embed_mesh(X)

    w = pysal.lat2W(x, x)
    adj_mtx = np.array(w.full()[0])

    coords = []
    for i in range(x):
        for j in range(x):
            coords.append([i, j, 0])

    coords = np.array(coords)
    return adj_mtx, coords


def load_csv(x):
    """
    loads cached adj_mtx data
    :param x: size (only 8 and 24 work for the mnist examples.)
    :return:
    """
    if x == 8:
        result = genfromtxt('../mesh/mtx8.csv', delimiter=',')
        result[0][0] = 0
    elif x == 24:
        result = genfromtxt('../mesh/mtx24.csv', delimiter=',')
        result[0][0] = 0
    else:
        print("please 8 or 24, sizes compatible for the dataset. 'None' returned.")
        return None
    return result
