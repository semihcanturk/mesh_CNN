import autograd.numpy as np
import mesh_traversal

region = mesh_traversal.find_region(93, 3)
print(region)

lev0 = region[0]
lev1 = region[1:7]
lev2 = region[7:]
print(lev0)
print(lev1)
print(lev2)

center = 93
r = 1
strides = mesh_traversal.mesh_strider(region, center, r)
print("Strides: ")
print(strides)

np.random.seed(5)   # for generating consistent results
filters = np.random.rand(6,7)    # 6 filters, length 7 each

large_region = mesh_traversal.find_region(93, 20)
conv = mesh_traversal.mesh_convolve(large_region, filters)
print("Convolution Result: ")
print(conv)
