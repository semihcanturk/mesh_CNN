"""Convolutional neural net on MNIST, modeled on 'LeNet-5',
http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf"""
from __future__ import absolute_import
from __future__ import print_function
from builtins import range

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.signal
from autograd import grad
from mesh import mesh_traversal, generate_sphere_data, load_sphere
import mnist
import time
import datetime
import pickle

center = 0
r = 1
stride = 1
level = 2
order = None


class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

def make_batches(N_total, N_batch):
    start = 0
    batches = []
    while start < N_total:
        batches.append(slice(start, start + N_batch))
        start += N_batch
    return batches

def logsumexp(X, axis, keepdims=False):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=keepdims))

def make_nn_funs(input_shape, layer_specs, L2_reg):
    parser = WeightsParser()
    cur_shape = input_shape
    for layer in layer_specs:
        N_weights, cur_shape = layer.build_weights_dict(cur_shape)
        parser.add_weights(layer, (N_weights,))

    def predictions(W_vect, inputs):
        """Outputs normalized log-probabilities.
        shape of inputs : [data, color, y, x]"""
        cur_units = inputs
        for layer in layer_specs:
            cur_weights = parser.get(W_vect, layer)
            cur_units = layer.forward_pass(cur_units, cur_weights)
        return cur_units

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        log_lik = np.sum(predictions(W_vect, X) * T)
        return - log_prior - log_lik

    def frac_err(W_vect, X, T):
        return np.mean(np.argmax(T, axis=1) != np.argmax(pred_fun(W_vect, X), axis=1))

    return parser.N, predictions, loss, frac_err

class init_conv_layer(object):
    def __init__(self, kernel_shape, num_filters):
        self.kernel_shape = kernel_shape
        self.num_filters = num_filters

    def forward_pass(self, inputs, param_vector):
        # Input dimensions:  [data, color_in, y, x]
        # Params dimensions: [color_in, color_out, y, x]
        # Output dimensions: [data, color_out, y, x]
        params = self.parser.get(param_vector, 'params')    # filters
        biases = self.parser.get(param_vector, 'biases')
        biases = biases.reshape(biases.shape[0], biases.shape[1], 1)

        conv = mesh_traversal.mesh_convolve_tensorized(params, inputs)
        return conv + biases

    def build_weights_dict(self, input_shape):
        # Input shape : [color, y, x] (don't need to know number of data yet)
        self.parser = WeightsParser()
        self.parser.add_weights('params', (input_shape[0], self.num_filters)
                                          + self.kernel_shape)
        self.parser.add_weights('biases', (1, self.num_filters, 1, 1))
        output_shape = (self.num_filters,) + \
                       self.conv_output_shape(input_shape[1:], self.kernel_shape)
        return self.parser.N, output_shape

    def conv_output_shape(self, A, B):
        return A[0],

class conv_layer(object):
    def __init__(self, kernel_shape, num_filters):
        self.kernel_shape = kernel_shape
        self.num_filters = num_filters

    def forward_pass(self, inputs, param_vector):
        # Input dimensions:  [data, color_in, y, x]
        # Params dimensions: [color_in, color_out, y, x]
        # Output dimensions: [data, color_out, y, x]
        params = self.parser.get(param_vector, 'params')    # filters
        biases = self.parser.get(param_vector, 'biases')
        biases = biases.reshape(biases.shape[0], biases.shape[1], 1)
        if inputs.shape[2] == 162:
            adj_mtx = m2
            coords = np.array(v2)
            faces = f2
        elif inputs.shape[2] == 42:
            adj_mtx = m1
            coords = np.array(v1)
            faces = f1
        elif inputs.shape[2] == 12:
            adj_mtx = m0
            coords = np.array(v0)
            faces = f0
        conv = mesh_traversal.tensorize_and_convolve_mesh(params, adj_mtx, inputs, coords, r, stride)
        return conv + biases

    def build_weights_dict(self, input_shape):
        # Input shape : [color, y, x] (don't need to know number of data yet)
        self.parser = WeightsParser()
        self.parser.add_weights('params', (input_shape[0], self.num_filters)
                                          + self.kernel_shape)
        self.parser.add_weights('biases', (1, self.num_filters, 1, 1))
        output_shape = (self.num_filters,) + \
                       self.conv_output_shape(input_shape[1:], self.kernel_shape)
        return self.parser.N, output_shape

    def conv_output_shape(self, A, B):
        return A[0],

class maxpool_layer(object):
    def __init__(self, pool_shape):
        self.pool_shape = pool_shape

    def build_weights_dict(self, input_shape):
        # input_shape dimensions: [color, y, x]
        output_shape = list(input_shape)
        for i in [0]:
            assert input_shape[i + 1] % self.pool_shape[i] == 0, \
                "maxpool shape should tile input exactly"
            output_shape[i + 1] = int((input_shape[i + 1] + 6) / (self.pool_shape[i]-2))
        return 0, output_shape

    def forward_pass(self, inputs, param_vector):
        if inputs.shape[2] == 162:
            adj_mtx = m2
            coords = np.array(v2)
            order = o2
        elif inputs.shape[2] == 42:
            adj_mtx = m1
            coords = np.array(v1)
            order = o1
        elif inputs.shape[2] ==12:
            adj_mtx = m0
            coords = np.array(v0)
            order = o0
        new_shape = inputs.shape[:2]
        for i in [0]:
            pool_width = self.pool_shape[i]
            img_width = inputs.shape[i + 2]
            new_dim = int((img_width + 6) / (pool_width-2))
            new_shape += (new_dim,)
        result = None
        for i in range(new_dim):
            n = mesh_traversal.get_neighs(adj_mtx, coords, i, 1)
            nlist = None
            for neighbor in n:
                x = inputs[:, :, order.index(neighbor)]
                if nlist is None:
                    nlist = np.expand_dims(x, axis=2)
                else:
                    x = np.expand_dims(x, axis=2)
                    nlist = np.concatenate((nlist, x), axis=2)
            subresult = np.mean(nlist, axis=2)
            if result is None:
                result = np.expand_dims(subresult, axis=2)
            else:
                subresult = np.expand_dims(subresult, axis=2)
                result = np.concatenate((result, subresult), axis=2)
        return result

class full_layer(object):
    def __init__(self, size):
        self.size = size

    def build_weights_dict(self, input_shape):
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        self.parser = WeightsParser()
        self.parser.add_weights('params', (input_size, self.size))
        self.parser.add_weights('biases', (self.size,))
        return self.parser.N, (self.size,)

    def forward_pass(self, inputs, param_vector):
        params = self.parser.get(param_vector, 'params')
        biases = self.parser.get(param_vector, 'biases')
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        return self.nonlinearity(np.dot(inputs[:, :], params) + biases)

class tanh_layer(full_layer):
    def nonlinearity(self, x):
        try:
            return np.tanh(x)
        except:
            x = np.array(x)
            s = x.shape
            fin = np.empty([s[0], s[1]])
            for i in range(s[0]):
                for j in range(s[1]):
                    val_temp = x._value[i][j]
                    if isinstance(val_temp, np.float):
                        val = val_temp
                    elif isinstance(val_temp._value, np.float):
                        val = val_temp._value
                    elif isinstance(val_temp._value._value, np.float):
                        val = val_temp._value._value
                    else:
                        if i == 0:
                            val = val_temp._value._value._value._value
                        else:
                            val = val_temp._value._value._value._value._value._value._value._value
                    fin[i][j] = val
            return np.tanh(fin)

class softmax_layer(full_layer):
    def nonlinearity(self, x):
        return x - logsumexp(x, axis=1, keepdims=True)


if __name__ == '__main__':
    # Network parameters
    L2_reg = 1.0
    input_shape = (1, 162, 7,)
    layer_specs = [init_conv_layer((7,), 6),
                   maxpool_layer((6,)),
                   conv_layer((7,), 2),
                   maxpool_layer((6,)),
                   tanh_layer(120),
                   #tanh_layer(84),
                   softmax_layer(3)]

    # Training parameters
    param_scale = 0.9
    learning_rate = 1e-4 * 8
    momentum = 0.9
    batch_size = 150
    num_epochs = 50

    # Load and process mesh data
    print("Loading training data...")
    one_hot = lambda x, K: np.array(x[:,None] == np.arange(K)[None, :], dtype=int)

    v0, f0, m0 = load_sphere.load(0)
    v1, f1, m1 = load_sphere.load(1)
    v2, f2, m2 = load_sphere.load(2)

    matrices = [m0, m1, m2]

    try:
        train_batch, train_labels, test_batch, test_labels, \
        adj_mtx, mesh_vals, coords, faces = pickle.load(open("data.pickle", "rb"))
    except (OSError, IOError) as e:
        train_images, train_labels, test_images, test_labels, \
        adj_mtx, mesh_vals, coords, faces = generate_sphere_data.generate(1000)
        coords = np.array(coords)

        train_images = np.expand_dims(train_images, axis=1) / 255
        test_images = np.expand_dims(test_images, axis=1) / 255

        train_batch = mesh_traversal.mesh_strider_batch(adj_mtx, train_images, coords, r, stride)
        test_batch = mesh_traversal.mesh_strider_batch(adj_mtx, test_images, coords, r, stride)

        pickle.dump((train_batch, train_labels, test_batch, test_labels,
                     adj_mtx, mesh_vals, coords, faces), open("data.pickle", "wb"))

    stime0 = time.time()

    order = mesh_traversal.traverse_mesh(coords, faces, center, stride)  # list of vertices, ordered
    rem = set(range(len(mesh_vals))).difference(set(order))
    order = order + list(rem)

    o2 = order
    o1 = mesh_traversal.traverse_mesh(np.array(v1), f1, center, stride)
    rem1 = set(range(42)).difference(set(o1))
    o1 = o1 + list(rem1)
    o0 = mesh_traversal.traverse_mesh(np.array(v0), f0, center, stride)
    rem0 = set(range(12)).difference(set(o0))
    o0 = o0 + list(rem0)

    train_labels = one_hot(train_labels, 3)
    test_labels = one_hot(test_labels, 3)
    N_data = train_batch.shape[0]

    # Make neural net functions
    N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(input_shape, layer_specs, L2_reg)
    loss_grad = grad(loss_fun)

    # Initialize weights
    rs = npr.RandomState(14)    # fix seed
    W = rs.randn(N_weights) * param_scale

    # Check the gradients numerically, just to be safe
    # quick_grad_check(loss_fun, W, (train_images[:50], train_labels[:50]))

    print("    Epoch      |    Train err  |   Test error  ")
    def print_perf(epoch, W):
        test_perf  = frac_err(W, test_batch, test_labels)
        train_perf = frac_err(W, train_batch, train_labels)
        print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))

    # Train with sgd
    batch_idxs = make_batches(N_data, batch_size)
    cur_dir = np.zeros(N_weights)

    stime = time.time()
    print(stime-stime0)
    for epoch in range(num_epochs):
        print_perf(epoch, W)
        if epoch == num_epochs - 1:
            etime = time.time()
            print(etime-stime)
        for idxs in batch_idxs:
            grad_W = loss_grad(W, train_batch[idxs], train_labels[idxs])
            cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
            W -= learning_rate * cur_dir