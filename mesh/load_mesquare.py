# Think about the adjacency matrix. First, enumerate the nodes (pixels) in a M x N ' \
# image going down first, then right (e.g., first pixel in the second column gets id M+1). Now, consider a general point k ' \
# (not in the boundary): it has edges with nodes (up) k-1, (down) k+1, (left) k-M, (right) k+M. Then, to build the full mesh,
# you just need iterate over i=1,...,M and j=1,...,N making  k=(i-1)*M+j, k taking in account the border conditions
# (i.e., when i==1, i==M, j==1, j==N) to not add the corresponding edges.
import numpy as np
from conv import convolution_impl, mnist
import pickle
import pysal


def load():
    # image dimensions
    M = 28
    N = 28

    # Load and process MNIST data
    print("Loading training data...")

    add_color_channel = lambda x : x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)

    ##############
    #mnist.init()
    train_images, train_labels, test_images, test_labels = mnist.load()

    train_images = train_images.reshape((train_images.shape[0], 1, 28, 28)) / 255.0
    test_images = test_images.reshape((test_images.shape[0], 1, 28, 28)) / 255.0



    #train_images = train_images.reshape(train_images.shape[0], 28, 28)
    #test_images = test_images.reshape(test_images.shape[0], 28, 28)

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
    #adj_mtx, coords, faces
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

            # embedd image pixel on vector
            #Y[k] = X[N, M]

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
    X = np.random.rand(x, x)

    #adj_mtx, _ = embed_mesh(X)

    w = pysal.lat2W(x, x)
    adj_mtx = np.array(w.full()[0])

    coords = []
    for i in range(x):
        for j in range(x):
            coords.append([i, j, 0])

    coords = np.array(coords)
    # adj_mtx, coords, faces
    return adj_mtx, coords