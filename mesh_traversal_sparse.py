import os
os.environ['ETS_TOOLKIT'] = 'qt4'
import openmesh as om
import autograd.numpy as np
import numpy as npo
import math
from numpy import linalg as LA
from numpy import genfromtxt
from scipy import sparse

mesh = om.TriMesh()


# API
def create_adj_mtx(coords_file, tri_file):
    if isinstance(coords_file, str) and isinstance(tri_file, str):
        coords = genfromtxt(coords_file, delimiter=',')  # these include the coordinates for each point
        triangles = genfromtxt(tri_file, delimiter=',')  # these include the vertices in each triangle
        triangles = triangles.astype(int)
    else:
        coords = coords_file
        triangles = tri_file

    verts = []
    for i in coords:
        verts.append(mesh.add_vertex(i))
    faces = []
    for i in triangles:
        faces.append(mesh.add_face(verts[i[0]], verts[i[1]], verts[i[2]]))
    adj_mtx = np.zeros(shape=(coords.shape[0], coords.shape[0]))
    for i in triangles:
        p1 = i[0]
        p2 = i[1]
        p3 = i[2]

        adj_mtx[p1][p2] = 1
        adj_mtx[p2][p1] = 1

        adj_mtx[p1][p3] = 1
        adj_mtx[p3][p1] = 1

        adj_mtx[p2][p3] = 1
        adj_mtx[p3][p2] = 1

    sparse_adj = sparse.csr_matrix(adj_mtx)
    return sparse_adj, coords, triangles


def get_dist(coords, v,j):
    coords1 = coords[v]
    coords2 = coords[j]
    return math.sqrt((coords1[0]-coords2[0])**2 + (coords1[1]-coords2[1])**2 + (coords1[2]-coords2[2])**2)


def get_order(adj_mtx, coords, ix_list, closest_ix):
    arr = []
    seen = []
    arr.append(closest_ix)
    seen.append(closest_ix)
    v = closest_ix
    # find the neighbor
    neigh_list = []

    list_of_lists = []
    nz = adj_mtx.tolil().rows

    for i in ix_list:
        neighs = nz[i]
        list_of_lists.append(neighs)

    ct = 0
    for i in list_of_lists:
        if closest_ix in i:
            neigh_list.append(ix_list[ct])
        ct += 1

    if len(arr) == len(ix_list):
        return arr

    while len(arr) != len(ix_list):
        if len(neigh_list) == 2:
            v1 = neigh_list[0]
            v2 = neigh_list[1]
            x1 = coords[v1]
            x2 = coords[v2]
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


def find_region(adj_mtx, coords, vertex, r):
    verts = list()

    # level_0
    verts.append(vertex)
    v = vertex

    # find closest point in level 1
    nz = adj_mtx.tolil().rows
    ix_list = nz[v]

    dists = []
    for j in ix_list:
        d = get_dist(coords, v, j)
        dists.append(d)
    ix_min = ix_list[dists.index(min(dists))]
    closest_ix = ix_min

    # levels_>=1
    for i in range(1, r + 1):
        # this is the closest vertex of the new level
        # find the ordering of the level
        arr = get_order(adj_mtx, coords, ix_list, closest_ix)
        verts = verts + arr
        # get next level: for each in ix_list, get neighbors that are not in <verts>, then add them to the new list
        next_list = []
        for j in ix_list:
            new_row = nz[j]
            for k in new_row:
                if k not in verts:
                    next_list.append(k)
        next_list = list(set(next_list))

        # find starting point of next level using line eq
        c1 = coords[vertex]
        c2 = coords[closest_ix]
        line_dists = []
        for j in next_list:
            c3 = coords[j]
            line_dist = LA.norm(np.cross(c2 - c1, c1 - c3)) / LA.norm(c2 - c1)  # not exactly sure of this
            line_dists.append(line_dist)
        ix_list = next_list
        closest_ix = next_list[line_dists.index(min(line_dists))]
    return verts


def traverse_mesh(coords, triangles, center, stride=1):
    adj_mtx, coords, triangles = create_adj_mtx(coords, triangles)

    if stride == 1:
        vertex = center
        verts = list()

        # level_0
        verts.append(vertex)
        v = vertex

        # find closest point in level 1
        nz = adj_mtx.tolil().rows
        ix_list = nz[v]
        dists = []
        for j in ix_list:
            d = get_dist(coords, v, j)
            dists.append(d)
        ix_min = ix_list[dists.index(min(dists))]
        closest_ix = ix_min

        # levels_>=1
        while len(verts) != adj_mtx.shape[0]:    # until all vertices are seen
            # this is the closest vertex of the new level
            # find the ordering of the level
            arr = get_order(adj_mtx, coords, ix_list, closest_ix)
            verts = verts + arr
            # get next level: for each in ix_list, get neighbors that are not in <verts>, then add them to the new list
            next_list = []
            for j in ix_list:
                new_row = nz[j]
                for k in new_row:
                    if k not in verts:
                        next_list.append(k)
            next_list = list(set(next_list))
            if len(next_list) == 0:
                continue
            # find starting point of next level using line eq
            c1 = coords[vertex]
            c2 = coords[closest_ix]
            line_dists = []
            for j in next_list:
                c3 = coords[j]
                line_dist = LA.norm(np.cross(c2 - c1, c1 - c3)) / LA.norm(c2 - c1)  # not exactly sure of this
                line_dists.append(line_dist)
            ix_list = next_list
            closest_ix = next_list[line_dists.index(min(line_dists))]
        return verts

    else:   # multiple stride case

        vertex = center
        verts = list()

        # level_0
        verts.append(vertex)
        v = vertex

        seen = list()
        seen.append(v)

        nz = adj_mtx.tolil().rows
        ix_list = nz[v]
        dists = []
        for j in ix_list:
            d = get_dist(coords, v, j)
            dists.append(d)
        ix_min = ix_list[dists.index(min(dists))]
        closest_ix = ix_min

        add_to_verts = False
        ctr = 1
        # levels_>=1
        while len(seen) != adj_mtx.shape[0]:  # until all vertices are seen
            # this is the closest vertex of the new level
            # find the ordering of the level
            arr = get_order(adj_mtx, coords, ix_list, closest_ix)
            seen = seen + arr

            if add_to_verts:    # add only every other level to the traversal list
                temp_arr = arr[::stride]
                verts = verts + temp_arr
            # get next level: for each in ix_list, get neighbors that are not in <verts>, then add them to the new list
            ctr = ctr + 1
            if ctr % stride == 0:
                add_to_verts = True
            else:
                add_to_verts = False
            next_list = []

            for j in ix_list:
                nz = adj_mtx.tolil().rows
                new_row = nz[j]

                for k in new_row:
                    if k not in seen:
                        next_list.append(k)
            next_list = list(set(next_list))
            if len(next_list) == 0:
                continue
            # find starting point of next level using line eq
            c1 = coords[vertex]
            c2 = coords[closest_ix]
            line_dists = []
            for j in next_list:
                c3 = coords[j]
                line_dist = LA.norm(np.cross(c2 - c1, c1 - c3)) / LA.norm(c2 - c1)  # not exactly sure of this
                line_dists.append(line_dist)
            ix_list = next_list
            closest_ix = next_list[line_dists.index(min(line_dists))]
        return verts


def mesh_strider(adj_mtx, coords, center, radius, max_radius = 3):
    # max_radius: the radius covering the whole mesh
    patches = []
    vertices = find_region(adj_mtx, coords, center, max_radius-1)   # list of vertices, ordered
    for v in vertices:                             # -1 because edges cant be centers
        patches.append(find_region(adj_mtx, coords, v, radius))     # so no patches with them as centers
    return patches


def mesh_convolve(filters, adj_mtx, coords):
    center = 93 # arbitrary
    r = 1   # arbitrary
    max_radius = 20 # this should be derived from the mesh itself
    strided_mesh = mesh_strider(adj_mtx, coords, center, r, max_radius=max_radius)
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


