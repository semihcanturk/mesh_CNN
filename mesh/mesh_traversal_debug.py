import os
os.environ['ETS_TOOLKIT'] = 'qt4'
import openmesh as om
import autograd.numpy as np
import numpy as npo
import math
from numpy import linalg as LA
from numpy import genfromtxt
import time
from scipy import sparse


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
            neighs = np.nonzero(adj_mtx[i])
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
        # until all vertices are seen #TODO: Fix inequality bug
        while len(verts) <= 0.9985 * l:
            # this is the closest vertex of the new level
            # find the ordering of the level
            if verbose_ctr == 130:
                print("Here we are!")
                print(len(verts))
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


def find_region(adj_mtx, coords, vertex, r):
    # TODO: Account for values - what are the values that we obtain from each vertex?
    # We should return the values from the vertices, not the vertices themselves in the final implementation.
    """
    Given a center and radius, calculates the traversal list of all vertices in a given depth radius
    :param adj_mtx: adjacency matrix of the mesh
    :param coords: coordinates of the vertices
    :param vertex: center vertex
    :param r: radius of region in terms of depth
    :return: traversal list of the vertices in the region
    """
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
        arr = get_order(adj_mtx, coords, ix_list, closest_ix)
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


def mesh_strider(adj_mtx, coords, faces, center, radius, stride):
    # TODO: Account for edge cases where we try to get a patch from an edge vertex, which shouldn't be possible.
    # Instead, traversal should stop when the edge of the patch reaches the edge of the mesh.
    # TODO: Account for values - what are the values that we obtain from each vertex?
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
    for v in vertices:                             # -1 because edges cant be centers
        patches.append(find_region(adj_mtx, coords, v, radius))     # so no patches with them as centers
    return patches


def mesh_convolve(filters, adj_mtx, coords, faces, center, r, stride):
    """
    Strides the mesh and applies a convolution to the patches
    :param filters: list of filters
    :param adj_mtx: adjacency matrix
    :param coords: coordinates of each vertex
    :return: result of the convolution operation
    """
    #center = 93 # arbitrary
    #r = 1   # arbitrary
    strided_mesh = mesh_strider(adj_mtx, coords, faces, center, r, stride)
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
