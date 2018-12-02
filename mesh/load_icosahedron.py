import openmesh as om
import autograd.numpy as np
import numpy as npo
import math

"""
Creates mesh data for an icosahedron.
"""

# vertices with ids 0:11 have 5 neighbors (causing 5-bug), all others 6

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

om.write_mesh('../data/icosahedron.off', mesh)


def load():
    return coords, faces