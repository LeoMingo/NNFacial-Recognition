#Adding L2
#+weight decay
#+tanh and ReLU
#cost and accuracy data accumulator


import numpy as np

import random
from os import times

class QuadraticCost(object):
    @staticmethod
    def cost(a, y):
        return 0.5*np.linalg.norm(a-y)**2
    """
    def cost_prime(self, a, y):
        return a-y
    """
    @staticmethod
    def delta(a, y, z):
        (a-y)*sigmoid_prime(z)

class NN(object):
    def __init__(self, sizes, output_rsts, cost=QuadraticCost):
        self.sizes = sizes
        self.layer_num = len(sizes)
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for y, x in zip(self.sizes[1:], self.sizes[:-1])]
        self.cost = cost

        if len(output_rsts) != self.sizes[-1]:
            print "Warning: The length of output_rsts   \
                   does not match the size of the last layer"
        olm_tups = [(idx,rst) for idx in xrange(self.sizes[-1])
                              for rst in output_rsts]
        self.output_layer_map = dict(olm_tups)


    def rst_idx_vec(self, rst):
        olm = self.output_layer_map
        keys = olm.keys
        values = olm.values
        idx = key(olm, rst)
        rst_idx_vec = np.zeros((len(olm), 1))
        rst_idx_vec[idx] = 1.0
        return rst_idx_vec



    def SGD(self, training_data, mini_batch_size=1,
                  epochs=30, eta=1.0, lmbda=1.0,
                  test_data=None, descending_eta=False, weight_decay=False,
                  graph_analyze=False):

        mbs = mini_batch_size
        training_data_len = len(training_data)

        training_accuracy = []
        training_cost = []
        test_accuracy = []
        test_cost = []
        sgd_start_time = times()[-1]
        for e in range(epochs):
            epoch_start_time = times()[-1]

            random.shuffle(training_data)
            mini_batches = [mb for mb in training_data[0:len(training_data):mbs]]
            for mb in mini_batches:
                self.update_nn(mb, eta, mbs, lmbda, training_data_len,
                               descending_eta, weight_decay)
            print "Epoch time spent: {0}s".format(times()[-1]-epoch_start_time)

            if test_data:
                print "test preciseness: {0}".format(self.eval_test(test_data))

            #if graph_analyze

        print "SGD Training finished after {0} epochs. \nTotal time spent: {1}s.".format(epochs, times()[-1]-sgd_start_time)


    def update_nn(self, mini_batch, eta, mini_batch_size, lmbda, n, descending_eta=False, weight_decay=False):
        mbs = mini_batch_size
        if descending_eta:
            eta = eta/1+d
        input_layer_data = mini_batch[0]
        ildrst = self.rst_idx_vec(mini_batch[1])
        nws,nbs = self.backprop(input_layer_data, ildrst)
        if weight_decay:
            self.weights = [(1-lmbda*eta/n)*w-nw*(eta/mbs) for w, nw in zip(self.weights, nws)]
        else:
            self.weights = [w-nw*(eta/mbs) for w, nw in zip(self.weights, nws)]
        self.biases = [b-nb*(eta/mbs) for b, nb in zip(self.biases, nbs)]

    def backprop(self, data, rst):
        nws = [np.zeros(w.shape) for w in self.weights]
        nbs = [np.zeros(b.shape) for b in self.biases]
        for i in self.weights:
            print i.shape
        for j in nws:
            print j.shape


        activation = data
        activations = [activation]
        zs = []
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (self.cost).delta(self.feedforward(data), rst, zs[-1])
        #delta = cost_prime(rst, self.feedforward(data))*sigmoid_prime(zs[-1])
        delta = self.feedforward(data)*sigmoid_prime(zs[-1])-rst
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


"""
Probably need a bunch of wrapper classes to wrap the following functions up
"""

def tanh(z):
     return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
def ReLU(z):
    #^----------------^
    return np.argmax(0, z)

def sigmoid_prime(z):
    return sigmoid(z)*(1-np.exp(-z))



def key(dic, val):
    k=None
    keys = dic.keys()
    values = dic.values()
    for i in xrange(len(values)):
        if values[i] == val:
            k = i
    return k



