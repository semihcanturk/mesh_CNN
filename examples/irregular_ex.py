import numpy as np
import math
import openmesh as om
from mesh import mesh_traversal_deprecated


coords = np.array([
    [0, 0, 15],
    [-0.5, 0.87, 10],
    [-1.0,  0, 10],
    [-0.5, -0.87, 10],
    [0.5, -0.87, 10],
    [1, 0, 10],
    [0.5, 0.87, 10],
    [0, 0, 5],
], dtype=float)

faces = np.array([
    [0, 1, 2],
    [0, 2, 3],
    [0, 3, 4],
    [0, 4, 5],
    [0, 5, 6],
    [0, 6, 1],
    [1, 2, 7],
    [2, 3, 7],
    [3, 4, 7],
    [4, 5, 7],
    [5, 6, 7],
    [6, 1, 7]
])

mesh = om.TriMesh()

vlist = []
for i in coords:
    vlist.append(mesh.add_vertex(i))

flist = []
for i in faces:
    flist.append(mesh.add_face(vlist[i[0]], vlist[i[1]], vlist[i[2]]))

om.write_mesh('../data/irreg.off', mesh)


def load():
    return coords, faces


one_stride_s = mesh_traversal_deprecated.traverse_mesh(coords, faces, 0, is_sparse=True)
two_stride_s = mesh_traversal_deprecated.traverse_mesh(coords, faces, 0, 2, is_sparse=True)
three_stride_s = mesh_traversal_deprecated.traverse_mesh(coords, faces, 0, 3, is_sparse=True)

print("SPARSE MATRIX IMPLEMENTATION")
print("Result of 1-stride example: {}".format(one_stride_s))
print("Result of 2-stride example: {}".format(two_stride_s))
print("Result of 3-stride example: {}".format(three_stride_s))