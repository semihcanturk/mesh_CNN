import numpy as np
from mesh import load_sphere, mesh_traversal_debug
from scipy import sparse


def generate():
    verts, faces, adj_mtx = load_sphere.load(2)
    #adj_mtx, _, _ = mesh_traversal_debug.create_adj_mtx(np.array(verts), np.array(faces), is_sparse=True)
    print(sparse.issparse(adj_mtx))

    n_patterns = 5

    mesh_data = list()
    for i in range(100):
        mesh_vals = np.random.choice(np.array([20, 40, 60, 80, 100]), size=162, replace=True)
        if i < 20:
            for j in range(n_patterns):
                target0 = np.random.choice(range(162), 1)
                target1 = mesh_traversal_debug.find_region(adj_mtx, mesh_vals, np.array(verts), target0[0], 1)
                target1 = target1[1:]
                for j in target0:
                    mesh_vals[j] = 100
                for j in target1:
                    mesh_vals[j] = 60
        mesh_data.append(mesh_vals)

    X = np.vstack(mesh_data)
    ones = np.full(20, 1)
    zeros = np.zeros(80)
    y = np.append(ones, zeros)
    y = y.reshape(100,1)
    y = y.astype(int)
    data = np.concatenate((X, y), axis=1)
    np.random.shuffle(data)
    train = data[:80]
    test = data[80:]

    train_data = train[:, :162]
    train_labels = train[:, 162]
    test_data = test[:, :162]
    test_labels = test[:, 162]

    return train_data, train_labels, test_data, test_labels, adj_mtx, mesh_vals, verts, faces

