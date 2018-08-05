# ConvCNN Documentation

## convolution_impl.py
- - - -
### as_strided_seq(b, patch, stride=1)
Creates a view into the array with the given shape and strides.

~**Parameters**~
* **b:** Input array to be strided.  Assumes that the array is 4-dimensional (x, y, z, z), where y=1 and the last two dimensions forming a square.
* **patch:** Length of one side of the square “patch”. For example, in the MNIST dataset example, each image is 28x28, and is partitioned into 5x5 patches, with a patch parameter of 5. Patch size essentially should be the same as the filter size of the convolutional neural network (CNN). This parameter must be smaller than z.
* **stride:** The number of pixels skipped in each iteration, both horizontally and vertically. In the current implementation, this parameter is reliable only for 1, which is default. See *Current/Potential Issues* for more information.

~**Returns**~
* **p:** The strided array. The dimensions of the array will be (x, y, k, k, z-k+1, z-k+1), where k is equal to the *patch* parameter.

~**Current/Potential Issues**~
* In the current implementation, the *stride* parameter is reliable only for 1, which is default. For any other cases, padding should be implemented in order to not run into dimension mismatch issues.
- - - -
### convolve_seq(a, b)
Computes the discrete linear convolution of two arrays.

~**Parameters**~
* **a:** First input array. In context of CNNs, this array includes the filters.
* **b:** Second input array. In context of CNNs, this array includes image data.
* 
~**Returns**~
* **p:** The convoluted array. The dimensions are (x, q, z-k+1, z-k+1) where q is the number of filters from a,
- - - -
### random_example()
Runs a basic convolution example using randomly generated arrays of appropriate dimensions.

~**Returns**~
* **p:** The convoluted array that is produced using the *as_strided_seq()* and *convolve_seq()* functions.

- - - -
### mnist_example()
Runs a basic convolution example using A and B arrays from the MNIST dataset.

~**Returns**~
* **p:** The convoluted array that is produced using the *as_strided_seq()* and *convolve_seq()* functions.