import os
os.environ['ETS_TOOLKIT'] = 'qt4'
import openmesh as om
from numpy import genfromtxt
import autograd.numpy as np
import numpy as npo
import mesh_traversal

mesh = om.TriMesh()

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

center = 93  # id of the center vertex, arbitrary
radius = 10  # desired radius, the algorithm will perform a BFS of this depth
patch = mesh_traversal.find_region(center, radius)  # the patch to be visualised on the brain
