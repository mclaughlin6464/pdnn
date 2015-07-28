#My tweak to read in my data types.

import cPickle
import gzip
import os
import sys, re
import glob

import numpy
import theano
import theano.tensor as T
from utils.utils import string_2_bool
from io_func.model_io import log
from io_func import smart_open, preprocess_feature_and_label, shuffle_feature_and_label
from io_func.pickle_io import PickleDataRead

class PickleDataRead2(PickleDataRead):

    def __init__(self, pfile_path_list, read_opts):

        super(PickleDataRead2, self).__init__(pfile_path_list, read_opts)

    def load_next_partition(self, shared_xy):
        pfile_path = self.pfile_path_list[self.cur_pfile_index]
        if self.feat_mat is None or len(self.pfile_path_list) > 1:

            fopen = smart_open(pfile_path, 'rb')
            self.feat_mat, self.label_mat = cPickle.load(fopen)

            fopen.close()
            shared_x, shared_y = shared_xy

            #TODO no longer label_vec, is array
            self.feat_mat, self.label_mat = \
                preprocess_feature_and_label(self.feat_mat, self.label_mat, self.read_opts)
            if self.read_opts['random']:
                shuffle_feature_and_label(self.feat_mat, self.label_mat)

            shared_x.set_value(self.feat_mat.astype(theano.config.floatX), borrow=True)
            #TODO types wrong here? Maybe?
            shared_y.set_value(self.label_mat.astype(theano.config.floatX), borrow=True)

        self.cur_frame_num = len(self.feat_mat)

        self.cur_pfile_index += 1

        if self.cur_pfile_index >= len(self.pfile_path_list):   # the end of one epoch
            self.end_reading = True
            self.cur_pfile_index = 0

    def make_shared(self):
        # define shared variables
        feat = numpy.zeros((10,10), dtype=theano.config.floatX)
        label = numpy.zeros((10,10), dtype=theano.config.floatX)

        shared_x = theano.shared(feat, name = 'x', borrow = True)
        shared_y = theano.shared(label, name = 'y', borrow = True)
        return shared_x, shared_y


