"""
mesh_patch_client.py
This client script creates an adjacency matrix from existing data, and given a desired center and radius, returns the list
of vertices that are within the specified depth radius. In the final implementation, find_region will return the values
of the vertices instead of their IDs.
"""
_author_ = "Semih Cantürk"
_credits_ = "Cassiano Becker"

from mesh import mesh_traversal

adj_mtx, coords, faces = mesh_traversal.create_adj_mtx('./data/data0.csv', './data/data1.csv')

center = 93  # id of the center vertex, arbitrary
radius = 2  # desired radius, the algorithm will perform a BFS of this depth
patch = mesh_traversal.find_region(adj_mtx, coords, center, radius)  # the patch to be visualised on the brain
print(patch)
