
# coding: utf-8

# In[1]:
import os
os.environ['ETS_TOOLKIT'] = 'qt4'
import openmesh as om
import autograd.numpy as np
import numpy as npo
import math
from numpy import linalg as LA

# In[2]:
mesh = om.TriMesh()

# In[3]:
from numpy import genfromtxt
data0 = genfromtxt('data0.csv', delimiter=',')
data1 = genfromtxt('data1.csv', delimiter=',')

# In[4]:
data1 = data1.astype(int)

# In[5]:
verts = []
for i in data0:
    verts.append(mesh.add_vertex(i))

# In[6]:
faces = []
for i in data1:
    faces.append(mesh.add_face(verts[i[0]], verts[i[1]], verts[i[2]]))

# In[7]:
mesh.points()

# In[8]:
om.write_mesh('mesh.off', mesh)

# In[9]:
om.read_trimesh('mesh.off')

# In[10]:
print(data1)

# In[11]:
print(data0.shape)

# In[12]:
adj_mtx = np.zeros(shape=(32492, 32492))

# In[13]:
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


# In[15]
# API
def get_dist(v,j):
    coords1 = data0[v]
    coords2 = data0[j]
    return math.sqrt((coords1[0]-coords2[0])**2 + (coords1[1]-coords2[1])**2 + (coords1[2]-coords2[2])**2)


# In[16]
def get_order(ix_list, closest_ix):
    arr = []
    seen = []
    arr.append(closest_ix)
    seen.append(closest_ix)
    v = closest_ix
    # find the neighbor
    neigh_list = []

    list_of_lists = []
    for i in ix_list:
        neighs = np.nonzero(adj_mtx[i])
        neighs = neighs[0]
        list_of_lists.append(neighs)

    ct = 0
    for i in list_of_lists:
        if closest_ix in i:
            neigh_list.append(ix_list[ct])
        ct += 1

    while (len(arr) != len(ix_list)):
        if len(neigh_list) == 2:
            v1 = neigh_list[0]
            v2 = neigh_list[1]
            x1 = data0[v1]
            x2 = data0[v2]
            if x1[0] <= x2[0]:
                v = v1
                arr.append(v)
                seen.append(v)

                neigh_list = []
                ct = 0
                for i in list_of_lists:
                    if v in i and ix_list[ct] not in seen:
                        neigh_list.append(ix_list[ct])
                        seen.append(ix_list[ct])
                    ct += 1
            else:
                v = v2
                arr.append(v)
                seen.append(v)

                neigh_list = []
                ct = 0
                for i in list_of_lists:
                    if v in i and ix_list[ct] not in seen:
                        neigh_list.append(ix_list[ct])
                        seen.append(ix_list[ct])
                    ct += 1

        if len(neigh_list) == 1:
            v = neigh_list[0]
            arr.append(v)
            seen.append(v)

            neigh_list = []
            ct = 0
            for i in list_of_lists:
                if v in i and ix_list[ct] not in seen:
                    neigh_list.append(ix_list[ct])
                    seen.append(ix_list[ct])
                ct += 1

        if len(neigh_list) == 0:
            return arr


# In[17]:
def find_region(vertex, r):
    verts = []

    # level_0
    verts.append(vertex)
    v = vertex

    # find closest point in level 1
    row = adj_mtx[v]
    ix_list = np.nonzero(row)
    ix_list = ix_list[0]
    dists = []
    for j in ix_list:
        d = get_dist(v, j)
        dists.append(d)
    ix_min = ix_list[dists.index(min(dists))]
    closest_ix = ix_min

    # levels_>=1
    for i in range(1, r + 1):
        # this is the closest vertex of the new level
        # find the ordering of the level
        arr = get_order(ix_list, closest_ix)
        verts = verts + arr
        # get next level: for each in ix_list, get neighbors that are not in <verts>, then add them to the new list
        next_list = []
        for j in ix_list:
            new_row = adj_mtx[j]
            new_row = np.nonzero(new_row)
            new_row = new_row[0]

            for k in new_row:
                if k not in verts:
                    next_list.append(k)
        next_list = list(set(next_list))

        # find starting point of next level using line eq
        c1 = data0[vertex]
        c2 = data0[closest_ix]
        line_dists = []
        for j in next_list:
            c3 = data0[j]
            line_dist = LA.norm(np.cross(c2 - c1, c1 - c3)) / LA.norm(c2 - c1)  # not exactly sure of this
            line_dists.append(line_dist)
        ix_list = next_list
        closest_ix = next_list[line_dists.index(min(line_dists))]
    return verts


# In[18]:
def mesh_strider(mesh, center, radius, max_radius = 3):
    # max_radius: the radius covering the whole mesh
    patches = []
    vertices = find_region(center, max_radius-1)   # list of vertices, ordered
    for v in vertices:                             # -1 because edges cant be centers
        patches.append(find_region(v, radius))     # so no patches with them as centers
    return patches


# In[19]
def mesh_convolve(mesh, filters):
    center = 93 # arbitrary
    r = 1   # arbitrary
    max_radius = 20 # this should be derived from the mesh itself
    strided_mesh = mesh_strider(mesh, center, r, max_radius=max_radius)
    arr = []
    for f in filters:
        row = []
        for p in strided_mesh:
            temp = np.einsum('i,i->', f, p)
            row.append(temp)
        if len(arr) == 0:
            arr = npo.array([row])
            row = []
        else:
            arr = npo.vstack((arr, [row]))
            row = []
    return arr





