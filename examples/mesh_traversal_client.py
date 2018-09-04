"""
mesh_traversal_client.py
This client script creates an adjacency matrix from existing data, and given the appropriate coordinates and triangles
for a icosahedron mesh, traverses it and outputs the traversal order.
"""
_author_ = "Semih Cantürk"
_credits_ = "Cassiano Becker"

import os
os.environ['ETS_TOOLKIT'] = 'qt4'
import openmesh as om
import autograd.numpy as np
import numpy as npo
from mesh import mesh_traversal, mesh_traversal_sparse
import math
import time

# neighbor_count = adj_mtx.sum(axis=0)    # vertices with ids 0:11 have 5 neighbors (causing 5-bug), all others 6
# neighs_0 = npo.nonzero(adj_mtx[0])
# print(neighs_0) # [12, 68, 1720, 3372, 11800]

# here is an icosahedron, everyone's favorite platonic solid
r = (1.0 + math.sqrt(5.0)) / 2.0
coords = np.array([
    [-1.0,   r, 0.0],
    [ 1.0,   r, 0.0],
    [-1.0,  -r, 0.0],
    [ 1.0,  -r, 0.0],
    [0.0, -1.0,   r],
    [0.0,  1.0,   r],
    [0.0, -1.0,  -r],
    [0.0,  1.0,  -r],
    [  r, 0.0, -1.0],
    [  r, 0.0,  1.0],
    [ -r, 0.0, -1.0],
    [ -r, 0.0,  1.0],
], dtype=float)

faces = np.array([
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [5, 4, 9],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1],
])

mesh = om.TriMesh()

vlist = []
for i in coords:
    vlist.append(mesh.add_vertex(i))

flist = []
for i in faces:
    flist.append(mesh.add_face(vlist[i[0]], vlist[i[1]], vlist[i[2]]))

om.write_mesh('./data/icosahedron.off', mesh)

# Example with non-sparse matrix
start = time.time()
one_stride = mesh_traversal.traverse_mesh(coords, faces, 0)
two_stride = mesh_traversal.traverse_mesh(coords, faces, 0, 2)
three_stride = mesh_traversal.traverse_mesh(coords, faces, 0, 3)
end = time.time()

print("NON_SPARSE MATRIX IMPLEMENTATION")
print("Result of 1-stride example: {}".format(one_stride))
print("Result of 2-stride example: {}".format(two_stride))  # [0] = Level 0, [8, 2, 9] = Level 2, only even indices,
                                                            # [6, 4] discarded as stride = 2
print("Result of 3-stride example: {}".format(three_stride))  # [0] = Level 0, [3] = Level 3
print("Runtime: {}\n".format(end - start))

start = time.time()
one_stride_s = mesh_traversal_sparse.traverse_mesh(coords, faces, 0)
two_stride_s = mesh_traversal_sparse.traverse_mesh(coords, faces, 0, 2)
three_stride_s = mesh_traversal_sparse.traverse_mesh(coords, faces, 0, 3)
end = time.time()

print("SPARSE MATRIX IMPLEMENTATION")
print("Result of 1-stride example: {}".format(one_stride_s))
print("Result of 2-stride example: {}".format(two_stride_s))  # [0] = Level 0, [8, 2, 9] = Level 2, only even indices,
                                                            # [6, 4] discarded as stride = 2
print("Result of 3-stride example: {}".format(three_stride_s))  # [0] = Level 0, [3] = Level 3
print("Runtime: {}\n".format(end - start))
