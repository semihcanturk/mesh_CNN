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
import math
import datetime
import pickle

convolve = autograd.scipy.signal.convolve
from mesh import load_mesquare

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
        params = self.parser.get(param_vector, 'params')
        biases = self.parser.get(param_vector, 'biases')
        biases = biases.reshape(biases.shape[0], biases.shape[1], 1)

        a = int(math.sqrt(inputs.shape[2]))
        b = int(math.sqrt(params.shape[2]))
        inputs_temp = inputs.reshape(inputs.shape[0], inputs.shape[1], a, a)
        params_temp = params.reshape(params.shape[0], params.shape[1], b, b)

        conv = convolve(inputs_temp, params_temp, axes=([2, 3], [2, 3]), dot_axes = ([1], [0]), mode='valid')
        conv = conv.reshape(conv.shape[0], conv.shape[1], conv.shape[2] * conv.shape[2])
        return conv + biases

    def build_weights_dict(self, input_shape):
        # Input shape : [color, y, x] (don't need to know number of data yet)
        self.parser = WeightsParser()
        self.parser.add_weights('params', (input_shape[0], self.num_filters)
                                          + self.kernel_shape)
        self.parser.add_weights('biases', (1, self.num_filters, 1, 1))

        i1 = int(math.sqrt(input_shape[1]))
        new_i1 = (i1 - 4) ** 2
        output_shape = (self.num_filters,) + \
                       self.conv_output_shape([new_i1], self.kernel_shape)
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
            output_shape[i + 1] = int(input_shape[i + 1] / self.pool_shape[i])
        return 0, output_shape

    def forward_pass(self, inputs, param_vector):
        n = mesh_traversal.get_neighs_sq(inputs)
        result = np.array(n)
        result = np.moveaxis(result, 0, 2)

        # new_shape = inputs.shape[:2]
        # for i in [0]:
        #     pool_width = self.pool_shape[i]
        #     img_width = inputs.shape[i + 2]
        #     new_dim = int(img_width / (pool_width))
        #     new_shape += (new_dim,)
        # result = None
        # for i in range(new_dim):
        #
        #     if new_shape[2] == 16:
        #         #n = mesh_traversal.get_neighs(adj_mtx_2, coords_2, i, 1)
        #         n = mesh_traversal.get_neighs_sq(inputs)
        #     else:
        #         #n = mesh_traversal.get_neighs(adj_mtx, coords, i, 1)
        #         n = mesh_traversal.get_neighs_sq(inputs)
        #
        #     nlist = None
        #     for neighbor in n:
        #         #TODO: fix by applying small (8x8=64 vertex) order
        #         if new_shape[2] == 16:
        #             x = inputs[:, :, o2.index(neighbor)]
        #         else:
        #             x = inputs[:, :, order.index(neighbor)]
        #         if nlist is None:
        #             nlist = np.expand_dims(x, axis=2)
        #         else:
        #             x = np.expand_dims(x, axis=2)
        #             nlist = np.concatenate((nlist, x), axis=2)
        #     subresult = np.max(nlist, axis=2)
        #     if result is None:
        #         result = np.expand_dims(subresult, axis=2)
        #     else:
        #         subresult = np.expand_dims(subresult, axis=2)
        #         result = np.concatenate((result, subresult), axis=2)
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
        return np.tanh(x)

class softmax_layer(full_layer):
    def nonlinearity(self, x):
        return x - logsumexp(x, axis=1, keepdims=True)


if __name__ == '__main__':
    # Network parameters
    L2_reg = 1.0
    input_shape = (1, 576, 25,)
    layer_specs = [init_conv_layer((25,), 6),
                   maxpool_layer((4,)),
                   conv_layer((25,), 16),
                   maxpool_layer((4,)),
                   tanh_layer(120),
                   tanh_layer(84),
                   softmax_layer(10)]

    # Training parameters
    param_scale = 0.1
    learning_rate = 1e-3
    momentum = 0.9
    batch_size = 150
    num_epochs = 50

    init_rate = learning_rate

    add_color_channel = lambda x : x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))
    one_hot = lambda x, K : np.array(x[:,None] == np.arange(K)[None, :], dtype=int)

    ##############

    train_batch, train_labels, test_batch, test_labels, adj_mtx, coords = load_mesquare.load()
    adj_mtx, _ = load_mesquare.create_mtx(28)  #TODO: check if correct

    order = mesh_traversal.traverse_mtx(adj_mtx, coords, center, stride)  # list of vertices, ordered
    rem = set(range(28 * 28)).difference(set(order))
    order = order + list(rem)

    adj_mtx_2, coords_2 = load_mesquare.create_mtx(8)
    o2 = mesh_traversal.traverse_mtx(adj_mtx_2, coords_2, center, stride)
    rem2 = set(range(8 * 8)).difference(set(o2))
    o2 = o2 + list(rem2)

    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
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

    #start = time.time()
    #sdate = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
    #print(sdate)

    stime = time.time()
    for epoch in range(num_epochs):
        print_perf(epoch, W)

        learning_rate = init_rate / (1 + epoch/5)

        if epoch == num_epochs - 1:
            etime = time.time()
            print(etime-stime)
        for idxs in batch_idxs:
            t1 = train_batch[idxs]
            t2 = train_labels[idxs]
            grad_W = loss_grad(W, t1, t2)
            cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
            W -= learning_rate * cur_dir