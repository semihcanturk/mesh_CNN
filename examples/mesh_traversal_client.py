import time
from mesh import mesh_traversal
from mesh import load_icosahedron

"""
mesh_traversal_client.py
This client script creates an adjacency matrix from existing data, and given the appropriate coordinates and triangles
for a icosahedron mesh, traverses it and outputs the traversal order.
"""
_author_ = "Semih Cantürk"
_credits_ = "Cassiano Becker"

coords, faces = load_icosahedron.load()

# Example with non-sparse matrix
start = time.time()
one_stride = mesh_traversal.traverse_mesh(coords, faces, 0, is_sparse=False)
two_stride = mesh_traversal.traverse_mesh(coords, faces, 0, 2, is_sparse=False)
three_stride = mesh_traversal.traverse_mesh(coords, faces, 0, 3, is_sparse=False)
end = time.time()

print("NON_SPARSE MATRIX IMPLEMENTATION")
print("Result of 1-stride example: {}".format(one_stride))
print("Result of 2-stride example: {}".format(two_stride))  # [0] = Level 0, [8, 2, 9] = Level 2, only even indices,
                                                            # [6, 4] discarded as stride = 2
print("Result of 3-stride example: {}".format(three_stride))  # [0] = Level 0, [3] = Level 3
print("Runtime: {}\n".format(end - start))

start = time.time()
one_stride_s = mesh_traversal.traverse_mesh(coords, faces, 0, is_sparse=True)
two_stride_s = mesh_traversal.traverse_mesh(coords, faces, 0, 2, is_sparse=True)
three_stride_s = mesh_traversal.traverse_mesh(coords, faces, 0, 3, is_sparse=True)
end = time.time()

print("SPARSE MATRIX IMPLEMENTATION")
print("Result of 1-stride example: {}".format(one_stride_s))
print("Result of 2-stride example: {}".format(two_stride_s))  # [0] = Level 0, [8, 2, 9] = Level 2, only even indices,
                                                            # [6, 4] discarded as stride = 2
print("Result of 3-stride example: {}".format(three_stride_s))  # [0] = Level 0, [3] = Level 3
print("Runtime: {}\n".format(end - start))
