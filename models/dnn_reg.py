'''
@Author Sean McLaughiln
This is my copy of the dnn module. I'm adding some features so I can use DNN for regression rather than just classification.
'''

import cPickle
import gzip
import os
import sys
import time
import collections

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from layers.regressions_sgd import Regression
from layers.mlp import HiddenLayer, DropoutHiddenLayer, _dropout_from_layer

from models.dnn import DNN

from io_func import smart_open
from io_func.model_io import _nnet2file, _file2nnet

class DNN_REG(DNN):

    def __init__(self, numpy_rng, theano_rng=None,
                 cfg = None,  # the network configuration
                 dnn_shared = None, shared_layers=[], input = None):

        super(DNN_REG, self).__init__(numpy_rng, theano_rng, cfg, dnn_shared, shared_layers, input)

        #A matrix now, not a vector
        self.y = T.matrix('y')
        if self.n_outs > 0:
            #remove logLayer
            self.layers.pop(-1)
            for i in xrange(len(self.logLayer.params)):
                self.params.pop(-1)
                self.delta_params.pop(-1)

        # We do not need this layer, so we have to remove it.
        self.regLayer = Regression(
                         input=self.layers[-1].output,
                         n_in=self.hidden_layers_sizes[-1], n_out=self.n_outs)
        if self.n_outs>0:
            #NOTE: Could just reassign in place rather than pop/append's
            self.layers.append(self.regLayer)
            self.params.extend(self.regLayer.params)
            self.delta_params.extend(self.regLayer.delta_params)

        #Redo these with the new layer
        self.finetune_cost = self.regLayer.negative_log_likelihood(self.y)
        self.errors = self.finetune_cost #without the regularization

        #NOTE: could just remove last layer and add new one.
        if self.l1_reg is not None:
            for i in xrange(self.hidden_layers_number):
                W = self.layers[i].W
                self.finetune_cost += self.l1_reg * (abs(W).sum())

        if self.l2_reg is not None:
            for i in xrange(self.hidden_layers_number):
                W = self.layers[i].W
                self.finetune_cost += self.l2_reg * T.sqr(W).sum()

    def build_finetune_functions(self, train_shared_xy, valid_shared_xy, batch_size):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')
        momentum = T.fscalar('momentum')

        # compute the gradients with respect to the model parameters

        #theano.printing.pydotprint(self.finetune_cost, outfile="finetune_cost.png", var_with_name_simple=True)

        gparams = T.grad(self.finetune_cost, self.params)

        #theano.printing.pydotprint(gparams, outfile="gparams.png", var_with_name_simple=True)

        # compute list of fine-tuning updates
        updates = collections.OrderedDict()
        for dparam, gparam in zip(self.delta_params, gparams):
            updates[dparam] = momentum * dparam - gparam*learning_rate
        for dparam, param in zip(self.delta_params, self.params):
            updates[param] = param + updates[dparam]

        if self.max_col_norm is not None:
            for i in xrange(self.hidden_layers_number):
                W = self.layers[i].W
                if W in updates:
                    updated_W = updates[W]
                    col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                    desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                    updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

        #theano.printing.pydotprint(self.errors, outfile="errors.png", var_with_name_simple=True)

        train_fn = theano.function(inputs=[index, theano.Param(learning_rate, default = 0.0001),
              theano.Param(momentum, default = 0.5)],
              outputs=self.errors,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        #theano.printing.pydotprint(train_fn , outfile="train_fn.png", var_with_name_simple=True)

        valid_fn = theano.function(inputs=[index],
              outputs=self.errors,
              givens={
                self.x: valid_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        return train_fn, valid_fn

    def build_finetune_functions_kaldi(self, train_shared_xy, valid_shared_xy):

        (train_set_x, train_set_y) = train_shared_xy
        (valid_set_x, valid_set_y) = valid_shared_xy

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.fscalar('learning_rate')
        momentum = T.fscalar('momentum')

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        #updates = collections.OrderedDict()
        updates = {}
        for dparam, gparam in zip(self.delta_params, gparams):
            updates[dparam] = momentum * dparam - gparam*learning_rate
        for dparam, param in zip(self.delta_params, self.params):
            updates[param] = param + updates[dparam]

        if self.max_col_norm is not None:
            for i in xrange(self.hidden_layers_number):
                W = self.layers[i].W
                if W in updates:
                    updated_W = updates[W]
                    col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                    desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                    updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

        train_fn = theano.function(inputs=[theano.Param(learning_rate, default = 0.0001),
              theano.Param(momentum, default = 0.5)],
              outputs=self.errors,
              updates=updates,
              givens={self.x: train_set_x, self.y: train_set_y})

        valid_fn = theano.function(inputs=[],
              outputs=self.errors,
              givens={self.x: valid_set_x, self.y: valid_set_y})

        return train_fn, valid_fn

