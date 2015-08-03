'''
@Author Sean McLaughiln
This is my copy of the logistic_sgd module. I'm adding some features so I can use DNN for regression rather than just classification.
'''

import cPickle
import gzip
import os
import sys
import time

import numpy

from layers.logistic_sgd import LogisticRegression

import theano
import theano.tensor as T

#It'd be possible to have a hidden layer with no activation, but I also need the negative_log_likelikhood function

class Regression(object):
    """ Class for multi-class logistic regression """

    def __init__(self, input, n_in, n_out, W=None, b=None):

        self.n_in = n_in
        self.n_out = n_out

        self.type = 'fc'#What's this?

        if W is None:
            W_values = numpy.zeros((n_in, n_out), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = W
        self.b = b

        self.delta_W = theano.shared(value = numpy.zeros_like(self.W.get_value(borrow=True)), name = 'delta_W')
        self.delta_b = theano.shared(value = numpy.zeros_like(self.b.get_value(borrow=True)), name = 'delta_b')
        # compute vector of class-membership probabilities in symbolic form

        self.output = T.dot(input, self.W) + self.b

        # compute prediction as class
        self.y_pred = self.output

        # parameters of the model
        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]

    def negative_log_likelihood(self, y):
        #TODO Change to a difference of vectors, as y will now be a label "array"
        return T.sum(T.mean((self.y_pred[T.arange(y.shape[0])]-y)**2, axis = 0))

