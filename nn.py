import numpy as np

import random
import os

class NN(object):
    def __init__(self, sizes, output_rsts):
        self.sizes = sizes
        self.layer_num = len(sizes)
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for y, x in zip(self.sizes[1:], self.sizes[:-1])]


        if len(output_rsts) != self.sizes[-1]:
            print "Warning: The length of output_rsts does not match the size of the last layer"
        olm_tups = [(idx,rst) for idx in xrange(self.sizes[-1])
                              for rst in output_rsts]
        self.output_layer_map = dict(olm)


    def rst_idx_vec(self, rst):
        olm = self.output_layer_map
        keys = olm.keys
        values = olm.values
        idx = key(olm, rst)
        rst_idx_vec = np.zeros((len(olm), 1))
        rst_idx_vec[idx] = 1.0
        return rst_idx_vec



    def SGD(self, train_data, mini_batch_size=1,
                  epochs=30, eta=1.0,
                  test_data=None, descending_eta=False):

        mbs = mini_batch_size

        sgd_start_time = os.times()[-1]
        for e in range(epochs):
            epoch_start_time = os.times()[-1]

            random.shuffle(train_data)
            mini_batches = [mb for mb in train_data[0:len(train_data):mbs]]
            for mb in mini_batches:
                self.update_nn(mb, eta, mbs, d_eta=descending_eta)
            print "Epoch time spent: {0}s".format(os.times()[-1]-epoch_start_time)

            if test_data:
                print "test preciseness: {0}".format(self.eval_test(test_data))

        print "SGD Training finished after {0} epochs. \nTotal time spent: {1}s.".format(epochs, os.times()[-1]-sgd_start_time)


    def update_nn(self, mini_batch, eta, mbs, d_eta=False):
        if d_eta:
            eta = eta/1+d

        input_layer_data = mini_batch[0]
        ildrst = self.rst_idx_vec(mini_batch[1])
        nws,nbs = self.backprop(input_layer_data, ildrst)
        self.weights = [w-nw*(eta/mbs) for w, nw in zip(self.weights, nws)]
        self.biases = [b-nb*(eta/mbs) for b, nb in zip(self.biases, nbs)]

    def backprop(self, data, rst):
        nws = [np.zeros(w.shape) for w in self.weights]
        nbs = [np.zeros(b.shape) for b in self.biases]


        activation = data
        activations = [activation]
        zs = []
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = cost_prime(rst, self.feedforward(data))*sigmoid_prime(zs[-1])
        nbs[-1] = delta
        #?---------ARRAY dot vs *---------?
        nws[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.layer_num):
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sigmoid_prime(zs[-l])
            nbs[-l] = delta
            nws[-l] = np.dot(delta, activations[-l+1].transpose())
        return (nws, nbs)



    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def eval_test(self, test_data):
        rst_arr = []
        for td in test_data:
            #^--------------------^#
            pred_rst_idx = np.argmax(self.feedforward(d[0]))
            ismatch = key(self.output_layer_map, d[1]) == pred_rst_idx
            rst_arr.append(ismatch)
        rst = 0
        for r in rst_arr:
            if r==True:
                rst+=1.0
        rst /= len(test_data)
        return rst



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-np.exp(-z))

def cost_prime(y, a):
    return y-a



def key(dic, val):
    keys = dic.keys
    values = dic.values
    for i in xrange(len(values)):
        if values[i] == val:
            key = i
    return key



