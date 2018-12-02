from __future__ import absolute_import
from __future__ import print_function
from builtins import range

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from mesh import mesh_traversal, load_sphere
import fmri_convnet.load_fmri as loader
import time
import datetime
import pickle
import h5py
import pandas as pd
from sklearn.metrics import confusion_matrix


center = 93
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
    batches_labels = []
    while start < N_total/4:
        batches.append(slice(start, start + N_batch))
        batches_labels.append(slice(start, (start + N_batch)))
        batches_labels.append(slice(start + int(N_total / 4), (start + int(N_total / 4) + N_batch)))
        batches_labels.append(slice(start + int(N_total / 2), (start + int(N_total / 2) + N_batch)))
        batches_labels.append(slice(start + int(3 * N_total / 4), (start + int(3 * N_total / 4) + N_batch)))
        start += N_batch
    return batches, batches_labels

def build_batch(idxs):
    with h5py.File('zero_train.h5', 'r') as hf:
        zero_train = hf['train'][:, idxs]
    with h5py.File('one_train.h5', 'r') as hf:
        one_train = hf['train'][:, idxs]
    with h5py.File('two_train.h5', 'r') as hf:
        two_train = hf['train'][:, idxs]
    with h5py.File('three_train.h5', 'r') as hf:
        three_train = hf['train'][:, idxs]

    tr_batch = np.concatenate((zero_train, one_train, two_train, three_train), axis=1)
    #tr_batch = np.expand_dims(tr_batch, axis=1)

    tr_batch = np.swapaxes(tr_batch, 1, 0)
    tr_batch = np.swapaxes(tr_batch, 1, 5)
    tr_batch = np.swapaxes(tr_batch, 4, 5)
    tr_batch = np.squeeze(tr_batch, axis=(2, 3))

    return tr_batch

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
        stimuli = ['none', 'LF', 'LH', 'RF', 'RH', 'T']

        y_test = np.argmax(T, axis=1)
        y_hat = np.argmax(pred_fun(W_vect, X), axis=1)

        C_mat = confusion_matrix(y_test, y_hat)
        C = pd.DataFrame(C_mat, index=stimuli, columns=stimuli)

        print(C)
        return np.mean(y_test != y_hat)

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

        conv = mesh_traversal.mesh_convolve_tensorized_dyn(params, inputs)
        return conv + biases

    def build_weights_dict(self, input_shape):
        # Input shape : [color, y, x] (don't need to know number of data yet)
        self.parser = WeightsParser()
        self.parser.add_weights('params', (input_shape[0], self.num_filters, input_shape[1])
                                          + self.kernel_shape)
        self.parser.add_weights('biases', (1, self.num_filters, 1, 1))
        output_shape = (self.num_filters,) + \
                       self.conv_output_shape(input_shape[2:], self.kernel_shape)
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
        conv = mesh_traversal.mesh_convolve_tensor(params, adj_mtx, inputs, coords, faces, center, r, stride)
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
        # if inputs.shape[2] == 162:
        #     adj_mtx = m2
        #     coords = np.array(v2)
        #     order = o2
        # elif inputs.shape[2] == 42:
        #     adj_mtx = m1
        #     coords = np.array(v1)
        #     order = o1
        # elif inputs.shape[2] ==12:
        #     adj_mtx = m0
        #     coords = np.array(v0)
        #     order = o0
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
        return np.tanh(x)

class softmax_layer(full_layer):
    def nonlinearity(self, x):
        return x - logsumexp(x, axis=1, keepdims=True)


if __name__ == '__main__':

    # Network parameters
    L2_reg = 1.0
    input_shape = (1, 15, 30886, 7)
    layer_specs = [init_conv_layer((7,), 100),
                   #maxpool_layer((6,)),

                   #conv_layer((7,), 2),
                   #maxpool_layer((6,)),

                   tanh_layer(120),
                   #tanh_layer(84),
                   softmax_layer(6)]

    # Training parameters
    param_scale = 0.1
    learning_rate = 1e-3
    momentum = 0.9
    batch_size = 150
    num_epochs = 50

    # Load and process mesh data
    print("Loading training data...")
    one_hot = lambda x, K: np.array(x[:,None] == np.arange(K)[None, :], dtype=int)

    v0, f0, m0 = load_sphere.load(0)
    v1, f1, m1 = load_sphere.load(1)
    v2, f2, m2 = load_sphere.load(2)

    #matrices = [m0, m1, m2]


    train_images, train_labels, test_images, test_labels, adj_mtx, coords, faces = loader.load()

    # train_labels = one_hot(train_labels, 6)
    # test_labels = one_hot(test_labels, 6)
    # N_data = len(train_labels)
    # batch_idxs, label_idx = make_batches(N_data, batch_size)
    # for i in range(len(batch_idxs)):
    #     train_batch_idx = build_batch(batch_idxs[i])
    #
    #     train_labels_idx_1 = train_labels[label_idx[i*4]]
    #     train_labels_idx_2 = train_labels[label_idx[i*4+1]]
    #     train_labels_idx_3 = train_labels[label_idx[i*4+2]]
    #     train_labels_idx_4 = train_labels[label_idx[i*4+3]]
    #
    #     train_labels_idx = train_labels_idx_1
    #     train_labels_idx = np.append(train_labels_idx, train_labels_idx_2, axis=0)
    #     train_labels_idx = np.append(train_labels_idx, train_labels_idx_3, axis=0)
    #     train_labels_idx = np.append(train_labels_idx, train_labels_idx_4, axis=0)



    #train_images, train_labels, test_images, test_labels, \
    #adj_mtx, mesh_vals, coords, faces = generate_sphere_data.generate(1000)
    coords = np.array(coords)

    #train_images = np.expand_dims(train_images, axis=1) #/ 255
    test_images = np.expand_dims(test_images, axis=1) #/ 255

    # try:
    #     with h5py.File('zero_train.h5', 'r') as hf:
    #         zero_train = hf['train'][:]
    #     with h5py.File('one_train.h5', 'r') as hf:
    #         one_train = hf['train'][:]
    #     with h5py.File('two_train.h5', 'r') as hf:
    #         two_train = hf['train'][:]
    #     with h5py.File('three_train.h5', 'r') as hf:
    #         three_train = hf['train'][:]
    # except:
    #     ct_train = train_images.shape[0]
    #     for i in range(4):
    #         data = train_images[(i*int(ct_train/4)):((i+1)*int(ct_train/4))]
    #         train_batch = mesh_traversal.mesh_strider_batch(adj_mtx, data, coords, faces, center, r, stride)
    #
    #         if i == 0:
    #             with h5py.File('zero_train.h5', 'w') as hf:
    #                 hf.create_dataset("train", data=train_batch)
    #         elif i == 1:
    #             with h5py.File('one_train.h5', 'w') as hf:
    #                 hf.create_dataset("train", data=train_batch)
    #         elif i == 2:
    #             with h5py.File('two_train.h5', 'w') as hf:
    #                 hf.create_dataset("train", data=train_batch)
    #         elif i == 3:
    #             with h5py.File('three_train.h5', 'w') as hf:
    #                 hf.create_dataset("train", data=train_batch)

    try:

        with h5py.File('zero_train.h5', 'r') as hf:
            train_batch_sample = hf['train'][:]
        with h5py.File('strided_test.h5', 'r') as hf:
            test_batch = hf['test'][:]
    except:
        test_batch = mesh_traversal.mesh_strider_batch(adj_mtx, test_images, coords, faces, center, r, stride)
        with h5py.File('strided_test.h5', 'w') as hf:
            hf.create_dataset("test", data=test_batch)

    # train_batches = [zero_train, one_train, two_train, three_train]
    #
    # for i in len(train_batches):
    #     train_batches[i] = np.swapaxes(train_batches[i], 1, 0)
    #     train_batches[i] = np.swapaxes(train_batches[i], 1, 5)
    #     train_batches[i] = np.swapaxes(train_batches[i], 4, 5)
    #     train_batches[i] = np.squeeze(train_batches[i], axis=(2, 3))

    train_batch_sample = np.swapaxes(train_batch_sample, 1, 0)
    train_batch_sample = np.swapaxes(train_batch_sample, 1, 5)
    train_batch_sample = np.swapaxes(train_batch_sample, 4, 5)
    train_batch_sample = np.squeeze(train_batch_sample, axis=(2, 3))

    test_batch = np.swapaxes(test_batch, 1, 0)
    test_batch = np.swapaxes(test_batch, 1, 5)
    test_batch = np.swapaxes(test_batch, 4, 5)
    test_batch = np.squeeze(test_batch, axis=(2, 3))


    stime0 = time.time()

    order = mesh_traversal.traverse_mesh(coords, faces, center, stride)  # list of vertices, ordered
    rem = set(range(adj_mtx.shape[0])).difference(set(order))
    order = order + list(rem)

    o2 = order
    #o1 = mesh_traversal.traverse_mesh(np.array(v1), f1, center, stride)
    #rem1 = set(range(42)).difference(set(o1))
    #o1 = o1 + list(rem1)
    #o0 = mesh_traversal.traverse_mesh(np.array(v0), f0, center, stride)
    #rem0 = set(range(12)).difference(set(o0))
    #o0 = o0 + list(rem0)

    train_labels = one_hot(train_labels, 6)
    test_labels = one_hot(test_labels, 6)
    N_data = len(train_labels)

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
        train_perf = frac_err(W, train_batch_sample, train_labels[:train_batch_sample.shape[0]])
        print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))

    # Train with sgd
    batch_idxs, label_idx = make_batches(N_data, batch_size)
    cur_dir = np.zeros(N_weights)

    #start = time.time()
    #sdate = datetime.datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
    #print(sdate)

    stime = time.time()
    print(stime-stime0)
    for epoch in range(num_epochs):
        print_perf(epoch, W)
        if epoch == num_epochs - 1:
            etime = time.time()
            print(etime-stime)

        for i in range(len(batch_idxs)):
            train_batch_idx = build_batch(batch_idxs[i])

            train_labels_idx_1 = train_labels[label_idx[i * 4]]
            train_labels_idx_2 = train_labels[label_idx[i * 4 + 1]]
            train_labels_idx_3 = train_labels[label_idx[i * 4 + 2]]
            train_labels_idx_4 = train_labels[label_idx[i * 4 + 3]]

            train_labels_idx = train_labels_idx_1
            train_labels_idx = np.append(train_labels_idx, train_labels_idx_2, axis=0)
            train_labels_idx = np.append(train_labels_idx, train_labels_idx_3, axis=0)
            train_labels_idx = np.append(train_labels_idx, train_labels_idx_4, axis=0)

            try:
                grad_W = loss_grad(W, train_batch_idx, train_labels_idx)
            except:
                print("Batch skipped")
            cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
            W -= learning_rate * cur_dir