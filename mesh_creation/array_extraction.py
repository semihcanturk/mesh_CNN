import os
import numpy as np
import nibabel as nib

file_gii = 'convdata/100206.L.very_inflated.32k_fs_LR.surf.gii'
file_gii = os.path.join(file_gii)
img = nib.load(file_gii)

img.print_summary()

# these are the spatial coordinates
data0 = img.darrays[0].data

# these are the mesh connections
data1 = img.darrays[1].data

np.savetxt("./data/data0.csv", data0, delimiter=",")
np.savetxt("./data/data1.csv", data1, delimiter=",")
