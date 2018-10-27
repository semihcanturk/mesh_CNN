import numpy as np
from mesh import load_sphere, mesh_traversal
from scipy import sparse


def generate(n_ex=5000):

    n_data = n_ex
    n_zeros = int(n_data * 0.5)
    n_ones = int(n_data * 0.5)
    train_test_split = 0.8

    mesh_data = list()
    for i in range(n_data):
        mesh_vals = np.ones((28, 28)) * -1 * 255
        if i < n_ones:
            x = np.random.choice(range(10))
            y = np.random.choice(range(24))
            mesh_vals[x, y] = np.random.uniform(50, 60)
            mesh_vals[x + 1, y] = np.random.uniform(50, 60)
            mesh_vals[x + 2, y] = np.random.uniform(50, 60)
            mesh_vals[x + 3, y] = np.random.uniform(50, 60)
            mesh_vals[x, y + 1] = np.random.uniform(50, 60)
            mesh_vals[x + 1, y + 1] = np.random.uniform(50, 60)
            mesh_vals[x + 2, y + 1] = np.random.uniform(50, 60)
            mesh_vals[x + 3, y + 1] = np.random.uniform(50, 60)
            mesh_vals[x, y + 2] = np.random.uniform(50, 60)
            mesh_vals[x + 1, y + 2] = np.random.uniform(50, 60)
            mesh_vals[x + 2, y + 2] = np.random.uniform(50, 60)
            mesh_vals[x + 3, y + 2] = np.random.uniform(50, 60)
            mesh_vals[x, y + 3] = np.random.uniform(50, 60)
            mesh_vals[x + 1, y + 3] = np.random.uniform(50, 60)
            mesh_vals[x + 2, y + 3] = np.random.uniform(50, 60)
            mesh_vals[x + 3, y + 3] = np.random.uniform(50, 60)
        # else:
        #     x = np.random.choice(range(14, 24))
        #     y = np.random.choice(range(24))
        #     mesh_vals[x, y] = 0
        #     mesh_vals[x + 1, y] = 0
        #     mesh_vals[x + 2, y] = 0
        #     mesh_vals[x + 3, y] = 0
        #     mesh_vals[x, y + 1] = 0
        #     mesh_vals[x + 1, y + 1] = 0
        #     mesh_vals[x + 2, y + 1] = 0
        #     mesh_vals[x + 3, y + 1] = 0
        #     mesh_vals[x, y + 2] = 0
        #     mesh_vals[x + 1, y + 2] = 0
        #     mesh_vals[x + 2, y + 2] = 0
        #     mesh_vals[x + 3, y + 2] = 0
        #     mesh_vals[x, y + 3] = 0
        #     mesh_vals[x + 1, y + 3] = 0
        #     mesh_vals[x + 2, y + 3] = 0
        #     mesh_vals[x + 3, y + 3] = 0
        mesh_data.append(mesh_vals)

    X = np.stack(mesh_data)
    X2 = np.reshape(X, (n_data, 784))
    ones = np.full(n_ones, 1)
    zeros = np.zeros(n_zeros)
    y = np.append(ones, zeros)
    y = y.reshape(n_data, 1)
    y = y.astype(int)
    data = np.concatenate((X2, y), axis=1)
    np.random.shuffle(data)

    y = data[:, 784]
    data = data[:, :784].reshape(n_data, 28, 28)

    n_train = int(n_data * train_test_split)
    train_data = data[:n_train]
    test_data = data[n_train:]
    train_labels = y[:n_train]
    test_labels = y[n_train:]

    return train_data, train_labels, test_data, test_labels

