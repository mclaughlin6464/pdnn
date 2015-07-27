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

class Regression(LogisticRegression):
    """ Class for multi-class logistic regression """

    def __init__(self, input, n_in, n_out, W=None, b=None):

        self.n_in = n_in
        self.n_out = n_out

        print n_in, n_out, 'Log'

        self.type = 'fc'

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
        #TODO no longer has meaning. Remove softmaxlogistic_sgd.py
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.output = self.p_y_given_x

        # compute prediction as class
        #TODO y_pred is a set of values, not an argmax
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]
        self.delta_params = [self.delta_W, self.delta_b]

    def negative_log_likelihood(self, y):
        #TODO Change to a difference of vectors, as y will now be a label "array"
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

