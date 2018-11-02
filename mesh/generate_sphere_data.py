import numpy as np
from mesh import load_sphere, mesh_traversal
from scipy import sparse


def generate(n_ex=5000):
    verts, faces, adj_mtx = load_sphere.load(2)
    #adj_mtx, _, _ = mesh_traversal_debug.create_adj_mtx(np.array(verts), np.array(faces), is_sparse=True)

    n_patterns = 5
    n_data = n_ex
    n_zeros = int(n_data * 0.3)
    n_ones = int(n_data * 0.3)
    n_twos = int(n_data * 0.4)
    train_test_split = 0.8

    np.random.seed(14)

    mesh_data = list()
    arr = [9, 8, 7, 1, 12, 3, 13, 0, 5, 18, 17, 19, 20, 6, 21, 22, 15, 4, 16, 14, 11, 10]
    for i in range(n_data):
        #mesh_vals = np.ones(162)
        mesh_vals = np.random.uniform(50, 150, size=162)
        if i < n_twos:
            for elt in arr:
                mesh_vals[elt] = np.random.uniform(100, 200)
        elif i < n_ones + n_twos:
            for j in range(n_patterns):
                target0 = np.random.choice(range(162), 1)
                _, target1 = mesh_traversal.find_region(adj_mtx, mesh_vals, np.array(verts), target0[0], 1, neighs=True)
                target1 = target1[1:]
                for k in target0:
                    mesh_vals[k] = np.random.uniform(100, 200)
                for k in target1:
                    mesh_vals[k] = np.random.uniform(100, 200)
        mesh_data.append(mesh_vals)

    X = np.vstack(mesh_data)
    twos = np.full(n_twos, 2)
    ones = np.full(n_ones, 1)
    zeros = np.zeros(n_zeros)
    y = np.concatenate((twos, ones, zeros))
    y = y.reshape(n_data, 1)
    y = y.astype(int)
    data = np.concatenate((X, y), axis=1)
    np.random.shuffle(data)

    n_train = int(n_data * train_test_split)
    train = data[:n_train]
    test = data[n_train:]

    train_data = train[:, :162]
    train_labels = train[:, 162]
    test_data = test[:, :162]
    test_labels = test[:, 162]

    return train_data, train_labels, test_data, test_labels, adj_mtx, mesh_data, verts, faces

