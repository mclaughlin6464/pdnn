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

class Regression(LogisticRegression):
    """ Class for multi-class logistic regression """

    def __init__(self, input, n_in, n_out, W=None, b=None):

        super(Regression, self).__init__(input, n_in, n_out, W, b)

        # compute vector of class-membership probabilities in symbolic form
        #linear activation
        self.output = T.dot(input, self.W) + self.b

        # compute prediction as class
        #y_pred is a set of values, not an argmax
        self.y_pred = self.output

    def negative_log_likelihood(self, y):
        #Change to a difference of vectors, as y will now be a label "array"
        return T.mean((self.y_pred-y)**2)


