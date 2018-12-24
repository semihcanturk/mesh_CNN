import os
#os.environ['ETS_TOOLKIT'] = 'qt4'
import openmesh as om
import autograd.numpy as np
import numpy as npo
import math
import autograd.numpy.linalg as LA
from numpy import genfromtxt
import time
from scipy import sparse
import itertools
import functools
import pickle

import h5py



# API
def create_adj_mtx(coords_file, tri_file, is_sparse=True):
    """
    Given a file with the coordinates for each point as well as triplets of ids for each mesh triangle, creates an
    adjacency matrix
    :param coords_file: a CSV file or python object that includes the coordinates for each point in the mesh
    :param tri_file: a CSV file or python object that includes the triplets of vertices for each triangle of the mesh
    :param is_sparse: whether a sparse implementation is desired
    :return: adjacency matrix and the coordinates and triangles parsed as python list objects
    """

    mesh = om.TriMesh()
    if isinstance(coords_file, str) and isinstance(tri_file, str):
        coords = genfromtxt(coords_file, delimiter=',')  # these include the coordinates for each point
        triangles = genfromtxt(tri_file, delimiter=',')  # these include the vertices in each triangle
        triangles = triangles.astype(int)
    else:
        coords = coords_file
        triangles = tri_file

    triangles = np.array(triangles)

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

    if is_sparse:
        return sparse.csr_matrix(adj_mtx), coords, triangles
    else:
        return adj_mtx, coords, triangles


def get_dist(coords, x,y):
    """
    Calculates the euclidean distance between two vertices in the mesh
    :param coords:
    :param x: vertex 1
    :param y: vertex 2
    :return: euclidean distance between x and y
    """
    coords1 = coords[x]
    coords2 = coords[y]
    return math.sqrt((coords1[0]-coords2[0])**2 + (coords1[1]-coords2[1])**2 + (coords1[2]-coords2[2])**2)


def get_order(adj_mtx, coords, ix_list, closest_ix, verts):
    """
    Given a list of vertices (at a certain depth away from the center), calculates traversal order
    :param adj_mtx: adjacency matrix of the mesh
    :param coords: list of coordinates for each vertex
    :param ix_list: list of vertices
    :param closest_ix: the starting vertex
    :return: ordered list of vertices
    """
    arr = []
    seen = set(verts)
    arr.append(closest_ix)
    seen.add(closest_ix)
    v = closest_ix
    # find the neighbor
    neigh_list = []

    list_of_lists = []

    if sparse.issparse(adj_mtx):
        nz = adj_mtx.tolil().rows
        for i in ix_list:
            neighs = nz[i]
            list_of_lists.append(neighs)
    else:
        for i in ix_list:
            try:
                neighs = np.nonzero(adj_mtx[i])
            except:
                print("boo")
            neighs = neighs[0]
            list_of_lists.append(neighs)

    ct = 0
    for i in list_of_lists:
        if closest_ix in i:
            if ix_list[ct] not in seen:
                neigh_list.append(ix_list[ct])
        ct += 1

    if len(arr) == len(ix_list):
        return arr

    while len(arr) != len(ix_list):
        if len(neigh_list) >= 2:
            v1 = neigh_list[0]
            v2 = neigh_list[1]
            x1 = coords[v1]
            x2 = coords[v2]
            if x1[0] <= x2[0]:
                v = v1
                arr.append(v)
                seen.add(v)

                neigh_list = []
                ct = 0
                for i in list_of_lists:
                    if v in i and ix_list[ct] not in seen:
                        neigh_list.append(ix_list[ct])
                        seen.add(ix_list[ct])
                    ct += 1
            else:
                v = v2
                arr.append(v)
                seen.add(v)

                neigh_list = []
                ct = 0
                for i in list_of_lists:
                    if v in i and ix_list[ct] not in seen:
                        neigh_list.append(ix_list[ct])
                        seen.add(ix_list[ct])
                    ct += 1

        if len(neigh_list) == 1:
            v = neigh_list[0]
            arr.append(v)
            seen.add(v)

            neigh_list = []
            ct = 0
            for i in list_of_lists:
                if v in i and ix_list[ct] not in seen:
                    neigh_list.append(ix_list[ct])
                    seen.add(ix_list[ct])
                ct += 1

        if len(neigh_list) == 0:
            return arr


def traverse_mtx(adj_mtx, coords, center, stride=1, verbose=False, is_sparse=True):
    """
    Calculates the traversal list of all vertices in the mesh
    :param coords: coordinates of the vertices
    :param faces: triplets of vertices for each triangle
    :param center: center vertex
    :param stride: the stride to be covered
    :param verbose: whether to print time after each iteration
    :param is_sparse: whether a sparse implementation is desired
    :return: list of all vertices in the mesh, starting from the center and in order of traversal
    """
    verbose_ctr = 1
    start = time.time()

    if stride == 1:
        vertex = center
        verts = list()

        # level_0
        verts.append(vertex)
        v = vertex

        # find closest point in level 1
        dists = []
        if sparse.issparse(adj_mtx):
            nz = adj_mtx.tolil().rows
            ix_list = nz[v]
        else:
            row = adj_mtx[v]
            ix_list = np.nonzero(row)
            ix_list = ix_list[0]

        for j in ix_list:
            d = get_dist(coords, v, j)
            dists.append(d)
        ix_min = ix_list[dists.index(min(dists))]
        closest_ix = ix_min

        # levels_>=1
        if sparse.issparse(adj_mtx):
            l = adj_mtx.shape[0]
        else:
            l = len(adj_mtx[0])
        # until all/certain percentage of vertices are seen
        while len(verts) <= 0.95 * l:
            # this is the closest vertex of the new level
            # find the ordering of the level
            if verbose_ctr == 130:
                print("Here we are!")
                print(len(verts))
            if verbose:
                print("Iteration {}: {}".format(verbose_ctr, time.time() - start))
            verbose_ctr = verbose_ctr + 1
            arr = get_order(adj_mtx, coords, ix_list, closest_ix, verts)

            a = set(ix_list)
            b = set(arr)
            d = a.difference(b)
            for e in d:
                arr.append(e)

            for i in arr:
                if i not in verts:
                    verts.append(i)
            # get next level: for each in ix_list, get neighbors that are not in <verts>, then add them to the new list
            next_list = []
            for j in ix_list:
                if sparse.issparse(adj_mtx):
                    new_row = nz[j]
                else:
                    new_row = adj_mtx[j]
                    new_row = np.nonzero(new_row)
                    new_row = new_row[0]

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

        if sparse.issparse(adj_mtx):
            nz = adj_mtx.tolil().rows
            ix_list = nz[v]
        else:
            row = adj_mtx[v]
            ix_list = np.nonzero(row)
            ix_list = ix_list[0]

        dists = []
        for j in ix_list:
            d = get_dist(coords, v, j)
            dists.append(d)
        ix_min = ix_list[dists.index(min(dists))]
        closest_ix = ix_min

        add_to_verts = False
        ctr = 1
        # levels_>=1
        if sparse.issparse(adj_mtx):
            l = adj_mtx.shape[0]
        else:
            l = len(adj_mtx[0])
        while len(seen) != l:  # until all vertices are seen
            # this is the closest vertex of the new level
            # find the ordering of the level
            arr = get_order(adj_mtx, coords, ix_list, closest_ix, seen)
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
                if sparse.issparse(adj_mtx):
                    nz = adj_mtx.tolil().rows
                    new_row = nz[j]
                else:
                    new_row = adj_mtx[j]
                    new_row = np.nonzero(new_row)
                    new_row = new_row[0]

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


def traverse_mesh(coords, faces, center, stride=1, verbose=False, is_sparse=True):
    """
    Calculates the traversal list of all vertices in the mesh
    :param coords: coordinates of the vertices
    :param faces: triplets of vertices for each triangle
    :param center: center vertex
    :param stride: the stride to be covered
    :param verbose: whether to print time after each iteration
    :param is_sparse: whether a sparse implementation is desired
    :return: list of all vertices in the mesh, starting from the center and in order of traversal
    """
    adj_mtx, coords, faces = create_adj_mtx(coords, faces, is_sparse)
    verbose_ctr = 1
    start = time.time()

    if stride == 1:
        vertex = center
        verts = list()

        # level_0
        verts.append(vertex)
        v = vertex

        # find closest point in level 1
        dists = []
        if sparse.issparse(adj_mtx):
            nz = adj_mtx.tolil().rows
            ix_list = nz[v]
        else:
            row = adj_mtx[v]
            ix_list = np.nonzero(row)
            ix_list = ix_list[0]

        for j in ix_list:
            d = get_dist(coords, v, j)
            dists.append(d)
        ix_min = ix_list[dists.index(min(dists))]
        closest_ix = ix_min

        # levels_>=1
        if sparse.issparse(adj_mtx):
            l = adj_mtx.shape[0]
        else:
            l = len(adj_mtx[0])
        # until all vertices are seen
        while len(verts) <= 0.95 * l:
            # this is the closest vertex of the new level
            # find the ordering of the level
            if verbose:
                print("Iteration {}: {}".format(verbose_ctr, time.time() - start))
            verbose_ctr = verbose_ctr + 1
            arr = get_order(adj_mtx, coords, ix_list, closest_ix, verts)
            for i in arr:
                if i not in verts:
                    verts.append(i)
            # get next level: for each in ix_list, get neighbors that are not in <verts>, then add them to the new list
            next_list = []
            for j in ix_list:
                if sparse.issparse(adj_mtx):
                    new_row = nz[j]
                else:
                    new_row = adj_mtx[j]
                    new_row = np.nonzero(new_row)
                    new_row = new_row[0]

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

        if sparse.issparse(adj_mtx):
            nz = adj_mtx.tolil().rows
            ix_list = nz[v]
        else:
            row = adj_mtx[v]
            ix_list = np.nonzero(row)
            ix_list = ix_list[0]

        dists = []
        for j in ix_list:
            d = get_dist(coords, v, j)
            dists.append(d)
        ix_min = ix_list[dists.index(min(dists))]
        closest_ix = ix_min

        add_to_verts = False
        ctr = 1
        # levels_>=1
        if sparse.issparse(adj_mtx):
            l = adj_mtx.shape[0]
        else:
            l = len(adj_mtx[0])
        while len(seen) != l:  # until all vertices are seen
            # this is the closest vertex of the new level
            # find the ordering of the level
            arr = get_order(adj_mtx, coords, ix_list, closest_ix, seen)
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
                if sparse.issparse(adj_mtx):
                    nz = adj_mtx.tolil().rows
                    new_row = nz[j]
                else:
                    new_row = adj_mtx[j]
                    new_row = np.nonzero(new_row)
                    new_row = new_row[0]

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


def find_region(adj_mtx, mesh_vals, coords, vertex, r, neighs=False):
    # We should return the values from the vertices, not the vertices themselves in the final implementation.
    """
    Given a center and radius, calculates the traversal list of all vertices in a given depth radius
    :param adj_mtx: adjacency matrix of the mesh
    :param coords: coordinates of the vertices
    :param vertex: center vertex
    :param r: radius of region in terms of depth
    :return: traversal list of the vertices in the region
    """

    @functools.lru_cache()
    def get_neighs(vertex, r):
        verts = list()

        # level_0
        verts.append(vertex)
        v = vertex

        # find closest point in level 1
        if sparse.issparse(adj_mtx):
            nz = adj_mtx.tolil().rows
            ix_list = nz[v]
        else:
            row = adj_mtx[v]
            ix_list = np.nonzero(row)
            ix_list = ix_list[0]

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
            arr = get_order(adj_mtx, coords, ix_list, closest_ix, verts)
            verts = verts + arr
            # get next level: for each in ix_list, get neighbors that are not in <verts>, then add them to the new list
            next_list = []
            for j in ix_list:
                if sparse.issparse(adj_mtx):
                    new_row = nz[j]
                else:
                    new_row = adj_mtx[j]
                    new_row = np.nonzero(new_row)
                    new_row = new_row[0]

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
                line_dist = LA.norm(np.cross(c2 - c1, c1 - c3)) / LA.norm(c2 - c1)  # calculate distance to line
                line_dists.append(line_dist)
            ix_list = next_list
            closest_ix = next_list[line_dists.index(min(line_dists))]
        return verts

    verts = get_neighs(vertex, r)

    vals = list()
    for i in verts:
        try:
            vals.append(mesh_vals[0, i])
        except:
            vals.append(mesh_vals[i])

    if neighs:
        return vals, verts
    else:
        return vals


def get_neighs_sq(inputs):
    verts = list()

    if inputs.shape[2] == 576:
        for i in range(12):
            for j in range(12):
                x = 2 * i + 48 * j
                a = inputs[:, :, x]
                b = inputs[:, :, x + 1]
                c = inputs[:, :, x + 24]
                d = inputs[:, :, x + 25]
                temp = np.stack((a, b, c, d))
                temp = np.amax(temp, axis=0)
                verts.append(temp)

    elif inputs.shape[2] == 64:
        for i in range(4):
            for j in range(4):
                x = 2 * i + 16 * j
                a = inputs[:, :, x]
                b = inputs[:, :, x + 1]
                c = inputs[:, :, x + 8]
                d = inputs[:, :, x + 9]
                temp = np.stack((a, b, c, d))
                temp = np.amax(temp, axis=0)
                verts.append(temp)

    return verts


def get_neighs_sq2(adj_mtx, i):
    new_row = adj_mtx[i]
    new_row = np.nonzero(new_row)
    verts = new_row[0]
    return verts


def get_neighs(adj_mtx, coords, vertex, r):
    verts = list()

    # level_0
    verts.append(vertex)
    v = vertex

    # find closest point in level 1
    if sparse.issparse(adj_mtx):
        nz = adj_mtx.tolil().rows
        ix_list = nz[v]
    else:
        row = adj_mtx[v]
        ix_list = np.nonzero(row)
        ix_list = ix_list[0]

    dists = []
    for j in ix_list:
        d = get_dist(coords, v, j)
        dists.append(d)
    ix_min = ix_list[dists.index(min(dists))]
    closest_ix = ix_min

    for i in range(1, r + 1):
        # this is the closest vertex of the new level
        # find the ordering of the level
        arr = get_order(adj_mtx, coords, ix_list, closest_ix, verts)
        verts = verts + arr
        # get next level: for each in ix_list, get neighbors that are not in <verts>, then add them to the new list
        next_list = []
        for j in ix_list:
            if sparse.issparse(adj_mtx):
                new_row = nz[j]
            else:
                new_row = adj_mtx[j]
                new_row = np.nonzero(new_row)
                new_row = new_row[0]

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
            line_dist = LA.norm(np.cross(c2 - c1, c1 - c3)) / LA.norm(c2 - c1)  # calculate distance to line
            line_dists.append(line_dist)
        ix_list = next_list
        closest_ix = next_list[line_dists.index(min(line_dists))]
    return verts


def mesh_strider(adj_mtx, mesh_vals, coords, faces, center, radius, stride):
    """
    Returns a list of patches after traversing and obtaining the patches for each mesh
    :param adj_mtx: adjacency matrix of the mesh
    :param coords:  coordinates of each vertes
    :param faces:   the vertices in each triangle of the mesh
    :param center:  center vertex
    :param radius:  radius for the patches
    :param stride:  stride in each traversal step
    :return:    array of patches in traversal order, in form of list of lists
    """
    patches = []
    vertices = traverse_mesh(coords, faces, center, stride)   # list of vertices, ordered
    rem = set(range(mesh_vals.shape[0])).difference(set(vertices))
    vertices = vertices + list(rem)
    for v in vertices:                             # -1 because edges cant be centers
        vals, neigh = find_region(adj_mtx, mesh_vals, coords, v, radius, neighs=True)
        patches.append(vals)     # so no patches with them as centers

    return patches


def mesh_strider_batch(adj_mtx, vals_list, coords, r, stride, cache=None, pool_type="max"):
    """
    Returns a list of patches after traversing and obtaining the patches for each mesh
    :param pool_type:
    :param cache:
    :param adj_mtx: adjacency matrix of the mesh
    :param coords:  coordinates of each vertes
    :param radius:  radius for the patches
    :param stride:  stride in each traversal step
    :return:    array of patches in traversal order, in form of list of lists
    """
    out = []
    #stime = time.time()
    # try:
    #     vertices = pickle.load(open("vertices.pkl", "rb"))
    # except:
    #     vertices = traverse_mesh(coords, faces, center, stride=stride, verbose=False, is_sparse=True)
    #     # If a full list is desiered, append non-traversed vertices to the end.
    #
    #     # rem = set(range(vals_list[0].shape[1])).difference(set(vertices))
    #     # vertices = vertices + list(rem)
    #
    #     with open('vertices.pkl', 'wb') as f:
    #         pickle.dump(vertices, f)

    #mtime = time.time()
    #print(mtime-stime)

    if cache is None:
        for v in range(vals_list.shape[2]):  # vertices:
            neighs = get_neighs(adj_mtx, coords, v, r)
            x = vals_list[:, :, [neighs], :]

            if len(neighs) < 7:
                temp = np.zeros((x.shape[0], x.shape[1], 1, 7 - len(neighs), x.shape[4]))
                x = np.append(x, temp, axis=3)
            elif len(neighs) > 7:
                x = x[:, :, :, :7, :]
            out.append(x)

        out = np.array(out)
        return out

    else:
        for v in range(vals_list.shape[2]):  # vertices:
            neighs = get_neighs(adj_mtx, coords, v, r)
            x = vals_list[:, :, [neighs], :]

            if len(neighs) < 7:
                temp = np.zeros((x.shape[0], x.shape[1], 1, 7 - len(neighs), x.shape[4]))
                x = np.append(x, temp, axis=3)
            elif len(neighs) > 7:
                x = x[:, :, :, :7, :]
            out.append(x)

        out = np.array(out)
        return out


def mesh_convolve(filters, adj_mtx, vals_list, coords, faces, center, r, stride):
    """
    Strides the mesh and applies a convolution to the patches. For testing purposes only, see versions below
    for efficient implementations.
    :param filters: list of filters
    :param adj_mtx: adjacency matrix
    :param coords: coordinates of each vertex
    :return: result of the convolution operation
    """

    f_count = vals_list.shape[1]
    conv_arr = []
    for vals in vals_list:
        depth_arr = []
        for c in range(f_count):
            strided_mesh = mesh_strider(adj_mtx, vals[c], coords, faces, center, r, stride)
            filter_arr = []
            for f in filters[c]:
                row = []
                for p in strided_mesh:
                    p = np.array(p)
                    try:
                        p = p / LA.norm(p)
                    except:
                        x = [i._value for i in p]
                        try:
                            p = x / LA.norm(x)
                        except:
                            y = [i._value for i in x]
                            try:
                                p = y / LA.norm(y)
                            except:
                                print("Convolution error.")
                    try:
                        temp = np.dot(f, p)
                        row.append(temp)
                    except:
                        temp = np.dot(f[:len(p)], p)
                        row.append(temp)
                if len(filter_arr) == 0:
                    filter_arr = np.array([row])
                else:
                    filter_arr = np.vstack((filter_arr, [row]))
            if len(depth_arr) == 0:
                depth_arr = np.array([filter_arr])
            else:
                depth_arr = np.vstack((depth_arr, [filter_arr]))
        if len(conv_arr) == 0:
            conv_arr = np.array([depth_arr])
        else:
            conv_arr = np.vstack((conv_arr, [depth_arr]))
    conv_arr = np.sum(conv_arr, axis=1)
    return conv_arr


def mesh_convolve_iter(a, adj_mtx, vals_list, coords, faces, center, r, stride):
    """
    Iteratively strides the mesh and applies a convolution to the patches. For testing purposes only, see tensorized
    versions below for efficient implementations.
    :param filters: list of filters
    :param adj_mtx: adjacency matrix
    :param coords: coordinates of each vertex
    :return: result of the convolution operation
    """

    a_dims = a.shape
    strided_mesh = mesh_strider_batch(adj_mtx, vals_list, coords, r, stride)
    mdi = MeshDataIterator(strided_mesh)

    out = []    #TODO: Consider dividing by norm
    for i in range(a_dims[1]):
        filters = []
        filter = a[:, i, :]
        mdi.reset()
        while mdi.has_next():
            patch = mdi.next()
            try:
                temp = npo.einsum('ij,ij->i', filter, patch)
            except:
                temp = npo.einsum('ij,ij->i', filter._value, patch)
                # patch = temp_b[ctr, j, k, :, :, :]
            filters.append(temp)
        if len(out) == 0:
            out = npo.array([filters])
        else:
            out = npo.vstack((out, [filters]))

    #out = out.reshape((a_dims[1], b.shape[0], b.shape[1], b.shape[2]))
    out = np.swapaxes(out, 0, 1)
    return out


def tensorize_and_convolve_mesh(a, adj_mtx, vals_list, coords, r, stride):
    """
    Strides the mesh and applies convolution operation. Prepares tensors within the function, so not as efficient as
    mesh_convolve_tensorized(). If operating on already strided data, use mesh_convolve_tensorized() or
    mesh_convolve_tensorized_dyn().
    :param filters: list of filters
    :param adj_mtx: adjacency matrix
    :param coords: coordinates of each vertex
    :return: result of the convolution operation
    """

    strided_mesh = mesh_strider_batch(adj_mtx, vals_list, coords, r, stride, None)
    try:
        out = npo.einsum(a, [0, 1, 2], strided_mesh, [3, 4, 2])
    except:
        try:
            a = a._value
            out = npo.einsum(a, [0, 1, 2], strided_mesh, [3, 4, 2])
        except:
            strided_mesh = strided_mesh._value
            out = npo.einsum(a, [0, 1, 2], strided_mesh, [3, 4, 2])

    out = out[0]
    out = np.swapaxes(out, 0, 1)
    return out

def tensorize_and_convolve_fmri(a, adj_mtx, vals_list, coords, r, stride):
    """
    Strides the mesh and applies convolution operation. Prepares tensors within the function, so not as efficient as
    mesh_convolve_tensorized(). If operating on already strided data, use mesh_convolve_tensorized() or
    mesh_convolve_tensorized_dyn().
    :param filters: list of filters
    :param adj_mtx: adjacency matrix
    :param coords: coordinates of each vertex
    :return: result of the convolution operation
    """

    vals_list = np.expand_dims(vals_list, axis=1)
    vals_list = np.swapaxes(vals_list, 2, 3)

    try:
        vals_list = vals_list._value
    except:
        pass

    strided_mesh = mesh_strider_batch(adj_mtx, vals_list, coords, r, stride, None)
    strided_vers = np.squeeze(np.array(strided_mesh))
    a = np.array([a])
    try:
        out = npo.einsum(a, [5, 3, 4, 2], strided_vers, [0, 1, 2, 3])
    except:
        try:
            a = a._value
            out = npo.einsum(a, [5, 3, 4, 2], strided_vers, [0, 1, 2, 3])
        except:
            strided_vers = strided_vers._value
            out = npo.einsum(a, [5, 3, 4, 2], strided_vers, [0, 1, 2, 3])

    #out = out[0]
    out = np.swapaxes(out, 0, 1)
    out = np.swapaxes(out, 1, 2)
    return out


def mesh_convolve_tensorized(a, b):
    """
    Performs convolution on the strided patches, used for the static data from load_mesquare.load()
    :param a: the filters, first array to be convoluted
    :param b: the strided mesh, second array to be convoluted
    :return: result of the convolution operation
    """

    try:
        out = npo.einsum(a, [0, 1, 2], b, [3, 4, 2])
    except:
        a = a._value
        out = npo.einsum(a, [0, 1, 2], b, [3, 4, 2])
    out = out[0]
    out = np.swapaxes(out, 0, 1)
    return out


def mesh_convolve_tensorized_dyn(a, b):
    """
    Performs convolution on the strided patches, used for the dynamic data from load_mesquare.load_dynamic()
    :param a: the filters, first array to be convoluted
    :param b: the strided mesh, second array to be convoluted
    :return: result of the convolution operation
    """

    try:
        out = npo.einsum(a, [0, 1, 5, 2], b, [3, 5, 4, 2])
    except:
        a = a._value
        out = npo.einsum(a, [0, 1, 5, 2], b, [3, 5, 4, 2])
    out = out[0]
    out = np.swapaxes(out, 0, 1)
    return out


class MeshDataIterator:
    """
    Iterator object to parse mesh data.
    """
    def __init__(self, b):
        self.ix = 0
        self.arr = b
        self.length = len(self.arr)

    def has_next(self):
        return False if self.ix >= self.length else True

    def next(self):
        #try:
        item = self.arr[self.ix]
        # except:
        #     print("boo")
        self.ix += 1
        return item

    def reset(self):
        self.ix = 0


def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    #assert os.path.exists(file)

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        assert lines[0] == 'OFF'

        parts = lines[1].split(' ')
        assert len(parts) == 3

        num_vertices = int(parts[0])
        assert num_vertices > 0

        num_faces = int(parts[1])
        assert num_faces > 0

        vertices = []
        for i in range(num_vertices):
            vertex = lines[2 + i].split(' ')
            vertex = [float(point) for point in vertex]
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[2 + num_vertices + i].split(' ')
            face = [int(index) for index in face]

            assert face[0] == len(face) - 1
            for index in face:
                assert index >= 0 and index < num_vertices

            assert len(face) > 1

            faces.append(face)

        return vertices, faces


def save_closest():
    v, _ = read_off('../fmri_convnet/mesh_2.off')
    coords, _ = read_off('../fmri_convnet/mesh_1.off')
    #v = np.array(v)
    #coords = np.array(coords)
    closest_list = []
    for i in v:
        dist = 100
        ix = None
        for j in coords:
            temp = (i[0]-j[0])**2 + (i[1]-j[1])**2 + (i[2]-j[2])**2
            if temp < dist:
                dist = temp
                ix =coords.index(j)#np.where(coords == j)[0][0]
        closest_list.append(ix)

    npo.savetxt("neighs_L2.csv", np.array(closest_list), delimiter=",")





