import numpy as np
from mesh import load_icosahedron, mesh_traversal
from scipy import sparse


def generate():
    """
    generates an icosahedron mesh representing a smaller brain mesh, and embeds random patterns
    drawn from a distribution, creating data for a basic discrimination task.
    :return: train/test and mesh data
    """
    verts, faces = load_icosahedron.load()
    adj_mtx, _, _ = mesh_traversal.create_adj_mtx(np.array(verts), np.array(faces), is_sparse=True)
    print(sparse.issparse(adj_mtx))

    n_patterns = 1
    n_data = 100
    mesh_data = list()
    for i in range(n_data):
        mesh_vals = np.random.choice(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), size=12, replace=True)
        if i < 20:
            for j in range(n_patterns):
                target0 = np.random.choice(range(12), 1)
                _, target1 = mesh_traversal.find_region(adj_mtx, mesh_vals, np.array(verts), target0[0], 1, neighs=True)
                target1 = target1[1:]
                for j in target0:
                    mesh_vals[j] = 0
                for j in target1:
                    mesh_vals[j] = 100
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

    train_data = train[:, :12]
    train_labels = train[:, 12]
    test_data = test[:, :12]
    test_labels = test[:, 12]

    return train_data, train_labels, test_data, test_labels, adj_mtx, mesh_vals, verts, faces