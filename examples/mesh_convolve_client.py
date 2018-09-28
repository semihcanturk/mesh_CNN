import autograd.numpy as np
import numpy as npo
import pickle
from mesh import mesh_traversal, mesh_traversal_debug, load_sphere
import random
from scipy import stats
from numpy import linalg as LA

"""
conv_debug.py
This client script creates an adjacency matrix from existing data, strides the mesh and convolves the result
with arbitrary data. These arbitrary filters (as well as the convolution values which are vertex IDs for now)
will be replaced in later implementations.
"""
_author_ = "Semih Cantürk"
_credits_ = "Cassiano Becker"

v, f = load_sphere.load()
v = np.array(v)
f = np.array(f)

# TODO: Verify that this is working correctly after implementation changes in mesh_traversal.py
adj_mtx, coords, faces = mesh_traversal_debug.create_adj_mtx(v, f)

#traversal_list = mesh_traversal_debug.traverse_mesh(coords, faces, 28105, verbose=True, is_sparse=True)
#pickle.dump(traversal_list, open("var.pickle", "wb"))
#print(len(traversal_list))


# the rest is the actual convolution

center = 93
radius = 1
stride = 1
# strides = mesh_traversal_debug.traverse_mesh(coords, faces, center)
# print("Strides: ")
# print(strides)

np.random.seed(5)   # for generating consistent results
filters = np.full((1, 19), 20)    # length 19 filter
filters[0][1:19:2] = 40
filters[0][0] = 100
filters[0][1:7] = 60

# filters = np.full((1, 19), 100)
norm = LA.norm(filters)
filters = filters / norm

mesh_vals = np.random.choice(np.array([20, 40, 60, 80, 100]), size=10242, replace=True)

target0 = [150]
target1 = [8833, 9091, 9127, 8864, 8866, 8831]
target2 = [2235, 9090, 2296, 9154, 2303, 9128, 2242, 8865, 2244, 8881, 2233, 8832]

for i in target0:
    mesh_vals[i] = 100
for i in target1:
    mesh_vals[i] = 60
ct = 0
for i in target2:
    if ct % 2 == 0:
        mesh_vals[i] = 40
    else:
        mesh_vals[i] = 20
    ct = ct + 1

# target = [150, 8833, 9091, 9127, 8864, 8866, 8831, 2235, 9090, 2296, 9154, 2303, 9128, 2242, 8865, 2244, 8881, 2233, 8832]
#
# for i in target:
#     mesh_vals[i] = 100

print(mesh_vals[150])

#mesh_vals = mesh_vals / norm

#large_region = mesh_traversal_debug.find_region(adj_mtx, coords, 93, 2)

#mesh_convolve(filters, adj_mtx, coords, faces, center, r, stride):
conv = mesh_traversal_debug.mesh_convolve(filters, adj_mtx, mesh_vals, coords, faces, 93, 2,
                                          1)  # filters, adj_mtx, coords, faces, center, r, stride
print("Convolution Result: ")
print(conv)
pickle.dump(conv, open("conv_result.pickle", "wb"))
