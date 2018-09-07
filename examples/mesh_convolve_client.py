import autograd.numpy as np
from mesh import mesh_traversal

"""
mesh_convolve_client.py
This client script creates an adjacency matrix from existing data, strides the mesh and convolves the result
with arbitrary data. These arbitrary filters (as well as the convolution values which are vertex IDs for now)
will be replaced in later implementations.
"""
_author_ = "Semih Cantürk"
_credits_ = "Cassiano Becker"

# TODO: Verify that this is working correctly after implementation changes in mesh_traversal.py
adj_mtx, coords, faces = mesh_traversal.create_adj_mtx('../data/data0.csv', '../data/data1.csv')

traversal_list = mesh_traversal.traverse_mesh(coords, faces, 93, verbose=True, is_sparse=True)
print(len(traversal_list))


# the rest is the actual convolution

# center = 93
# radius = 1
# stride = 1
# strides = mesh_traversal.mesh_strider(adj_mtx, coords, faces, center, radius, stride)
# print("Strides: ")
# print(strides)
#
# np.random.seed(5)   # for generating consistent results
# filters = np.random.rand(6,7)    # 6 filters, length 7 each
#
# large_region = mesh_traversal.find_region(93, 20)
# conv = mesh_traversal.mesh_convolve(large_region, filters)  # filters, adj_mtx, coords, faces, center, r, stride
# print("Convolution Result: ")
# print(conv)
