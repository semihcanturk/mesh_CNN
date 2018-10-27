import openmesh as om
from math import sqrt
from mesh import mesh_traversal
import numpy as np

"""

"""


def load(subdiv=2):
    # -----------------------------------------------------------------------------
    # Settings

    scale = 1
    #subdiv = 2

    # -----------------------------------------------------------------------------
    # Functions

    middle_point_cache = {}

    def vertex(x, y, z):
        """ Return vertex coordinates fixed to the unit sphere """

        length = sqrt(x ** 2 + y ** 2 + z ** 2)

        return [(i * scale) / length for i in (x, y, z)]

    def middle_point(point_1, point_2):
        """ Find a middle point and project to the unit sphere """

        # We check if we have already cut this edge first
        # to avoid duplicated verts
        smaller_index = min(point_1, point_2)
        greater_index = max(point_1, point_2)

        key = '{0}-{1}'.format(smaller_index, greater_index)

        if key in middle_point_cache:
            return middle_point_cache[key]

        # If it's not in cache, then we can cut it
        vert_1 = verts[point_1]
        vert_2 = verts[point_2]
        middle = [sum(i) / 2 for i in zip(vert_1, vert_2)]

        verts.append(vertex(*middle))

        index = len(verts) - 1
        middle_point_cache[key] = index

        return index

    # -----------------------------------------------------------------------------
    # Make the base icosahedron

    # Golden ratio
    PHI = (1 + sqrt(5)) / 2

    verts = [
        vertex(-1, PHI, 0),
        vertex(1, PHI, 0),
        vertex(-1, -PHI, 0),
        vertex(1, -PHI, 0),

        vertex(0, -1, PHI),
        vertex(0, 1, PHI),
        vertex(0, -1, -PHI),
        vertex(0, 1, -PHI),

        vertex(PHI, 0, -1),
        vertex(PHI, 0, 1),
        vertex(-PHI, 0, -1),
        vertex(-PHI, 0, 1),
    ]

    faces = [
        # 5 faces around point 0
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],

        # Adjacent faces
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],

        # 5 faces around 3
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],

        # Adjacent faces
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    # -----------------------------------------------------------------------------
    # Subdivisions

    for i in range(subdiv):
        faces_subdiv = []
        for tri in faces:
            v1 = middle_point(tri[0], tri[1])
            v2 = middle_point(tri[1], tri[2])
            v3 = middle_point(tri[2], tri[0])

            faces_subdiv.append([tri[0], v1, v3])
            faces_subdiv.append([tri[1], v2, v1])
            faces_subdiv.append([tri[2], v3, v2])
            faces_subdiv.append([v1, v2, v3])

        faces = faces_subdiv

    mesh = om.TriMesh()

    vlist = []
    for i in verts:
        vlist.append(mesh.add_vertex(i))

    flist = []
    for i in faces:
        flist.append(mesh.add_face(vlist[i[0]], vlist[i[1]], vlist[i[2]]))

    om.write_mesh('../data/small_sphere.off', mesh)

    adj_mtx, _, _ = mesh_traversal.create_adj_mtx(np.array(verts), np.array(faces), is_sparse=True)

    return verts, faces, adj_mtx