# Conv_CNN



### cnn
Temporary directory for fully working CNN implementations.

**cifar10_cnn.py:** A basic CNN implementation on the CIFAR10 dataset and Keras.

### conv
Directory for low-level CNN implementation code including higher-level architectures.

**stride_experiments.py, stride_scratch.py:** Scratch files for convolution implementation.

**convolution_impl:** Convolution API for visual recognition based on CNNs.

**hips_convnet:** CNN on MNIST implementation using autograd.

### data
Includes all data used or sample data to experiment with.

**HCP:** All fMRI data can be found in this directory.

**Pickle files:** Saved convolution array data.

**coords.csv, triangles.csv:** Coordinate/face transformation of brain surface mesh data.

**icosahedron.off:** Example mesh for traversal testing.

### examples
**get_patch_client.py:** Basic test for mesh_traversal.find_region() functionality.

**mesh_convolve_client.py:** Basic tests for mesh_traversal.traverse_mesh() and convolve() functionality.

**mesh_traversal_client.py:** Basic tests for sparse and non-sparse implementations of mesh_traversal.traverse_mesh()

**mesh_from_file.py:** Basic script that creates a mesh from surface data.

**mesh_overlay.py:** Basic script to set an activation overlay over a pysurfer mesh.

### mesh
**load_icosahedron.py:** Script to create a icosahedron mesh out of a list of vertex coordinates and a list of faces.

**mesh_traversal.py:** Traversal API for brain mesh.

### mesh_creation
**array_extraction.py:** Script to transform brain surface data to arrays that contain vertex coordinates and triplets for mesh faces.