import pickle
import numpy as np

"""
mesh_from_file.py
Loads an HCP surface file and creates a mesh from it.
"""

_author_ = "Semih Cantürk"
_credits_ = "Cassiano Becker"

import os
import nibabel as nib
import openmesh as om

file_gii = '../data/HCP/convdata/100206.L.very_inflated.32k_fs_LR.surf.gii'
file_gii = os.path.join(file_gii)
img = nib.load(file_gii)

img.print_summary()

# these are the spatial coordinates
data0 = img.darrays[0].data

# these are the mesh connections
data1 = img.darrays[1].data

mesh = om.TriMesh()

verts = []

for i in data0:
    verts.append(mesh.add_vertex(i))

v = []
faces = []
for i in data1:
    v.append(i[0])
    v.append(i[1])
    v.append(i[2])
    faces.append(mesh.add_face(verts[i[0]], verts[i[1]], verts[i[2]]))

v = set(v)
print(len(verts))
pickle.dump(v, open("verts.pickle", "wb"))
om.write_mesh('example_mesh.off', mesh)