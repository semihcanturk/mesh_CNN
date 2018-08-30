import os
os.environ['ETS_TOOLKIT'] = 'qt4'
import openmesh as om
import autograd.numpy as np
import mesh_traversal
import numpy as npo
import math
from numpy import linalg as LA
mesh = om.TriMesh()

from numpy import genfromtxt
data0 = genfromtxt('data0.csv', delimiter=',')  # these include the coordinates for each point
data1 = genfromtxt('data1.csv', delimiter=',')  # these include the vertices in each triangle

data1 = data1.astype(int)

verts = []
for i in data0:
    verts.append(mesh.add_vertex(i))

faces = []
for i in data1:
    faces.append(mesh.add_face(verts[i[0]], verts[i[1]], verts[i[2]]))

om.write_mesh('mesh.off', mesh)

adj_mtx = np.zeros(shape=(32492, 32492))

for i in data1:
    p1 = i[0]
    p2 = i[1]
    p3 = i[2]

    adj_mtx[p1][p2] = 1
    adj_mtx[p2][p1] = 1

    adj_mtx[p1][p3] = 1
    adj_mtx[p3][p1] = 1

    adj_mtx[p2][p3] = 1
    adj_mtx[p3][p2] = 1

neighbor_count = adj_mtx.sum(axis=0)    # vertices with ids 0:11 have 5 neighbors (causing 5-bug), all others 6
neighs_0 = npo.nonzero(adj_mtx[0])
print(neighs_0) # [12, 68, 1720, 3372, 11800]

patch_5_bug = mesh_traversal.find_region(1720, 3)
print(patch_5_bug)