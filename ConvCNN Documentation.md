# ConvCNN Documentation

## Architecture Overview: Square Images (MNIST)
* INPUT [28x28]: Holds raw values for square image.
* CONV1 [5x5x6]: Computes the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. Given 6 filters, this will result in a [6x24x24] output.
* MAXPOOL1 [2x2]: Downsamples the output of CONV1 along spatial dimensions, resulting in [6x12x12] output.
* CONV2 [5x5x16]: Performs the same operations as CONV1, to an input of [6x12x12] as opposed to [28x28]. This results in an output of [16x8x8] for each filter, leading to [6x16x8x8].
* MAXPOOL2 [2x2]: Downsamples the output of CONV2 along spatial dimensions, resulting in [6x16x4x4] output.
* TANH1: Fully connected, operates on MAXPOOL2, reduces it to a 1D output of [120].
* TANH2: Transforms the [120] output to [84], outputs to softmax.
* SOFTMAX: Computes and outputs class probabilities.

## Architecture Overview: Mesh
TBD

## API: convolution_impl.py
- - - -
### as_strided_seq(b, patch, stride=1)
Creates a view into the array with the given shape and strides.

**Parameters**
* **b:** Input array to be strided.  Assumes that the array is 4-dimensional (x, y, z, z), where y=1 and the last two dimensions forming a square.
* **patch:** Length of one side of the square “patch”. For example, in the MNIST dataset example, each image is 28x28, and is partitioned into 5x5 patches, with a patch parameter of 5. Patch size essentially should be the same as the filter size of the convolutional neural network (CNN). This parameter must be smaller than z.
* **stride:** The number of pixels skipped in each iteration, both horizontally and vertically. In the current implementation, this parameter is reliable only for 1, which is default. See *Current/Potential Issues* for more information.

**Returns**
* **p:** The strided array. The dimensions of the array will be (x, y, k, k, z-k+1, z-k+1), where k is equal to the *patch* parameter.
**Current/Potential Issues**
* In the current implementation, the *stride* parameter is reliable only for 1, which is default. For any other cases, padding should be implemented in order to not run into dimension mismatch issues.
- - - -
### convolve_seq(a, b)
Computes the discrete linear convolution of two arrays.

**Parameters**
* **a:** First input array. In context of CNNs, this array includes the filters.
* **b:** Second input array. In context of CNNs, this array includes image data.

**Returns**
* **p:** The convoluted array. The dimensions are (x, q, z-k+1, z-k+1) where q is the number of filters from a, and x, z and k are as defined in *as_strided_seq()*.
- - - -
### random_example()
Runs a basic convolution example using randomly generated arrays of appropriate dimensions.

**Returns**
* **p:** The convoluted array that is produced using the *as_strided_seq()* and *convolve_seq()* functions.

- - - -
### mnist_example()
Runs a basic convolution example using A and B arrays from the MNIST dataset.

**Returns**
* **p:** The convoluted array that is produced using the *as_strided_seq()* and *convolve_seq()* functions.

- - - -

## API: mesh_convolution.py
- - - -
### get_patch(center, radius)
Given a center point and an appropriate radius, returns a list of datapoints that can be traversed in order.

**Parameters**
* **a:** ID of the point that will be assigned as center.
* **b:** The radius of the patch. This can be thought as the minimum jumps required to get from the center to a point on the edge.

**Returns**
* **p:** A patch, which is an ordered list with the center as the first element.

- - - -

### get_next_patch(patch)
Given a patch (or center point), returns the next patch of points to be traversed after the input patch is traversed.

**Parameters**
* **patch:** The current patch that is traversed

**Returns**
* **p:** The next patch to be covered, which is an ordered list with the next center as the first element.

- - - -

### mesh_strider(mesh, center, radius)
Given a complete mesh, will traverse it in patches based on a given center to start on as well as a patch radius, and will return an appropriate strided view into the mesh.

**Parameters**
* **mesh:** The mesh to be strided
* **center:** Center point to start on, can default to an arbitrary value.
* **radius:** The patch radius to be used, can default to an appropriate value.
**Returns**
* **p:** The strided mesh, which will be a multidimensional array object.

- - - -

### mesh_convolve(a, b)
Computes the discrete linear convolution of two arrays.

**Parameters**
* **a:** First input array. In context of CNNs, this array includes the filters.
* **b:** Second input array. In context of CNNs, this array includes image data.

**Returns**
* **p:** The convoluted array.
