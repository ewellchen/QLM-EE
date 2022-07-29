# -*- coding: utf-8 -*-
# implement of QLM-EE-real

from keras.layers import Embedding, GlobalMaxPooling1D,GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute, add, subtract
import keras
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from layers import *

import math
import numpy as np

from keras import regularizers
import keras.backend as K

from models.BasicModel import BasicModel

class EntangledNNreal(BasicModel):

    def initialize(self):

        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.dim = self.opt.map_dimension
        self.amplitude_embedding = amplitude_embedding_layer(input_shape=None, input_dim=len(self.opt.alphabet),
                                                             embedding_dim=self.dim,seed = None,
                                                             trainable=True,
                                                             l2_reg=self.opt.amplitude_l2)
        self.l2_normalization = L2Normalization(axis=3)
        self.l2_norm = L2Norm(axis=2, keep_dims=False)
        self.ngram_value = [int(n_value) for n_value in str(self.opt.ngram_value).split(',')]
        # self.ngram = [NGram(n_value=n_value) for n_value in self.ngram_value]
        self.measure = [RealMeasurement_vector(units=int(m_value)) for m_value in str(self.opt.measurement_size).split(',')]
        self.hidden_neuron = [int(num_neuron) for num_neuron in str(self.opt.hidden_neuron).split(',')]
        self.dense_1, self.dense_2 = [], []
        self.l2normilization = L2Normalization(axis=2)
        self.tensor_product = []
        self.ngram = []
        for n_value in self.ngram_value:
            self.ngram.append(NGram(n_value=n_value))
            self.tensor_product.append(Real_tensor_product(ngram=n_value))
            if n_value == 1:
                self.dense_1.append(None)
                self.dense_2.append(None)
                # self.dense_3.append(None)
            else:
                if self.hidden_neuron[n_value-1] == 0:

                    dense_2 = Dense(pow(self.dim, n_value),
                                           use_bias=False,
                                           kernel_initializer='glorot_normal',
                                           seed = 666,
                                           kernel_constraint = None)
                                           # kernel_initializer='glorot_uniform'
                    self.dense_1.append(None)
                    self.dense_2.append(dense_2)
                    # self.dense_3.append(None)
                else:
                    dense_1 = Dense(self.hidden_neuron[n_value-1],
                                           activation=None,
                                           use_bias=False,
                                           kernel_initializer='glorot_normal',
                                           kernel_regularizer = regularizers.l2(self.opt.amplitude_l2),
                                           kernel_constraint=None)
                                           # kernel_initializer='glorot_uniform')
                    dense_2 = Dense(pow(self.dim, n_value),
                                           activation=None,
                                           use_bias=False,
                                           kernel_initializer='glorot_normal',
                                           kernel_regularizer=regularizers.l2(self.opt.amplitude_l2),
                                           kernel_constraint=None)
                                           # kernel_initializer='glorot_uniform',)
                    # dense_3 = ComplexDense(pow(self.dim, n_value),
                    #                        activation=None,
                    #                        use_bias=False,
                    #                        kernel_initializer='glorot_normal',
                    #                        kernel_regularizer=None,
                    #                        kernel_constraint=None,
                    #                        seed = None)
                    self.dense_1.append(dense_1)
                    self.dense_2.append(dense_2)
                    # self.dense_3.append(dense_3)

        self.dropout_probs = Dropout(self.opt.dropout_rate_probs)

    def __init__(self, opt):
        super(EntangledNNreal, self).__init__(opt)

    def build(self):
        self.probs = self.get_representation(self.doc)

    def get_representation(self, doc):
        probs_list = []
        for n_val in self.ngram_value:
            inputs = self.ngram[n_val-1](doc)
            word = self.l2_normalization(self.amplitude_embedding(inputs))
            if n_val == 1:
                entangled_word = self.tensor_product[n_val-1](word)

            else:
                separable_word = self.tensor_product[n_val-1](word)
                if self.hidden_neuron[n_val-1] == 0:
                    entangled_word = self.dense_2[n_val-1](word)
                else:
                    latent = self.dense_1[n_val-1](separable_word)
                    # [latent_real, latent_image] = self.dense_2[n_val - 1]([latent_real, latent_image])
                    entangled_word = self.dense_2[n_val - 1](latent)
                    entangled_word = self.l2normilization(entangled_word)
            probs_list.append(self.measure[n_val-1](entangled_word))

        # conbine the probablities
        if len(probs_list) > 1:
            self.probs = Concatenation(axis=-1)(probs_list)
        else:
            self.probs = probs_list[0]
        # pooling
        if self.opt.pooling_type == 'max':
            self.probs = GlobalMaxPooling1D()(self.probs)
        elif self.opt.pooling_type == 'average':
            self.probs = GlobalAveragePooling1D()(self.probs)
        elif self.opt.pooling_type == 'none':
            self.probs = Flatten()(self.probs)
        elif self.opt.pooling_type == 'max_col':
            self.probs = GlobalMaxPooling1D()(Permute((2, 1))(self.probs))
        elif self.opt.pooling_type == 'average_col':
            self.probs = GlobalAveragePooling1D()(Permute((2, 1))(self.probs))
        else:
            print('Wrong input pooling type -- The default flatten layer is used.')
            self.probs = Flatten()(self.probs)
        self.probs = self.dropout_probs(self.probs)
        return (self.probs)



if __name__ == "__main__":
    import keras
    from keras.layers import Input, Dense, Activation, Lambda
    import numpy as np
    from keras import regularizers
    from keras.models import Model
    import sys
    from params import Params
    from dataset import qa
    import keras.backend as K
    import units
    import itertools
    from loss import *
    from units import to_array
    from keras.utils import generic_utils
    import argparse
    import models.representation as models
    params = Params()
    config_file = 'config/local.ini'    # define dataset in the config
    params.parse_config(config_file)
    import dataset
    reader = dataset.setup(params)
    params = dataset.process_embedding(reader,params)
    from keras.layers import Embedding, GlobalMaxPooling1D,GlobalAveragePooling1D, Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute

    from keras.models import Model, Input, model_from_json, load_model
    from keras.constraints import unit_norm
    from complexnn import *
    import math
    import numpy as np
    
    from keras import regularizers
    import keras.backend as K
    class DottableDict(dict):
        def __init__(self, *args, **kwargs):
            dict.__init__(self, *args, **kwargs)
            self.__dict__ = self
            self.allowDotting()
        def allowDotting(self, state=True):
            if state:
                self.__dict__ = self
            else:
                self.__dict__ = dict()
            
    self = DottableDict()
    self.opt = params
    self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
    #############################################
    #This parameter should be passed from params
#        self.ngram = NGram(n_value = self.opt.ngram_value)
    self.ngram = [NGram(n_value = int(n_value)) for n_value in self.opt.ngram_value.split(',')]
    #############################################
    self.phase_embedding= phase_embedding_layer(self.opt.max_sequence_length, self.opt.lookup_table.shape[0], self.opt.lookup_table.shape[1], trainable = self.opt.embedding_trainable,l2_reg=self.opt.phase_l2)

    self.amplitude_embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table), self.opt.max_sequence_length, trainable = self.opt.embedding_trainable, random_init = self.opt.random_init,l2_reg=self.opt.amplitude_l2)
    self.l2_normalization = L2Normalization(axis = 3)
    self.l2_norm = L2Norm(axis = 3, keep_dims = False)
    self.weight_embedding = Embedding(self.opt.lookup_table.shape[0], 1, trainable = True)
    self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))  # activation="sigmoid",
    self.dropout_embedding = Dropout(self.opt.dropout_rate_embedding)
    self.dropout_probs = Dropout(self.opt.dropout_rate_probs)
    self.projection = ComplexMeasurement(units = self.opt.measurement_size)

    
    doc = self.doc
