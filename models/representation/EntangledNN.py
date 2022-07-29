# -*- coding: utf-8 -*-
# implement of QLM-EE

from keras.layers import Embedding, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Masking, Flatten, Dropout, \
    Activation, concatenate, Reshape, Permute, Add, Subtract, Average
import keras
from keras.models import Model, Input, model_from_json, load_model
from keras.constraints import unit_norm
from layers import *
from keras.initializers import RandomUniform,RandomNormal

import math
import numpy as np

from keras import regularizers
import keras.backend as K

from models.BasicModel import BasicModel


class EntangledNN(BasicModel):

    def initialize(self):

        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.dim = self.opt.map_dimension
        self.phase_embedding = phase_embedding_layer(input_shape=None, input_dim=len(self.opt.alphabet),
                                                     embedding_dim=self.dim, trainable=True,
                                                     l2_reg=self.opt.phase_l2)
        self.amplitude_embedding = amplitude_embedding_layer(input_shape=None, input_dim=len(self.opt.alphabet),
                                                             embedding_dim=self.dim,seed = None,
                                                             trainable=True,
                                                             l2_reg=self.opt.amplitude_l2)
        self.l2_normalization = L2Normalization(axis=3)
        self.ngram_value = [int(n_value) for n_value in str(self.opt.ngram_value).split(',')]
        # self.ngram = [NGram(n_value=n_value) for n_value in self.ngram_value]
        self.measure = [ComplexMeasurement_vector(units=int(m_value)) for m_value in str(self.opt.measurement_size).split(',')]
        self.hidden_neuron = [int(num_neuron) for num_neuron in str(self.opt.hidden_neuron).split(',')]
        self.dense_1, self.dense_2 = [], []
        self.complex_l2normilization = complex_L2Normalization(axis=(2,3))
        self.complex_tensor_product = []
        self.l2_norm = L2Norm(axis=3, keep_dims=False)
        self.l2_norm_complex = L2Norm_complex(axis=2, keep_dims=True)
        self.ngram = []
        for n_value in self.ngram_value:
            self.ngram.append(NGram(n_value=n_value))
            self.complex_tensor_product.append(Complex_tensor_product(ngram=n_value))
            if n_value == 1:
                self.dense_1.append(None)
                self.dense_2.append(None)
                # self.dense_3.append(None)
            else:
                if self.hidden_neuron[n_value-2] == 0:

                    dense_2 = ComplexDense(pow(self.dim, n_value),
                                           use_bias=False,
                                           kernel_initializer='glorot_normal',
                                           seed = 666,
                                           kernel_constraint = None)
                                           # kernel_initializer='glorot_uniform'
                    self.dense_1.append(None)
                    self.dense_2.append(dense_2)
                    # self.dense_3.append(None)
                else:
                    dense_1 = ComplexDense(self.hidden_neuron[n_value-2],
                                           activation=None,
                                           use_bias=False,
                                           kernel_initializer='glorot_normal',
                                           kernel_regularizer = regularizers.l2(self.opt.amplitude_l2),
                                           kernel_constraint=None,
                                           seed = None)
                                           # kernel_initializer='glorot_uniform')
                    dense_2 = ComplexDense(pow(self.dim, n_value),
                                           activation=None,
                                           use_bias=False,
                                           kernel_initializer='glorot_normal',
                                           kernel_regularizer=regularizers.l2(self.opt.amplitude_l2),
                                           kernel_constraint=None,
                                           seed = None)
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
        super(EntangledNN, self).__init__(opt)

    def build(self):
        self.probs = self.get_representation(self.doc)

    def get_representation(self, doc):
        probs_list = []
        count = 0
        for n_val in self.ngram_value:
            inputs = self.ngram[count](doc)
            word_phase = self.phase_embedding(inputs)
            amplitude = self.amplitude_embedding(inputs)
            # weight = Activation('softmax')(self.l2_norm(amplitude))
            word_amplitude = self.l2_normalization(amplitude)
            [word_real, word_image] = ComplexMultiply()([word_phase, word_amplitude])
            if n_val == 1:
                [entangled_real, entangled_image] = self.complex_tensor_product[count]([word_real, word_image])
                # weight = Activation('softmax')(self.l2_norm(amplitude))
            else:
                [separable_real, separable_image] = self.complex_tensor_product[count]([word_real, word_image])
                # [separable_real, separable_image] = self.complex_l2normilization([separable_real, separable_image])
                if self.hidden_neuron[count] == 0:
                    [entangled_real, entangled_image] = self.dense_2[count]([separable_real, separable_image])
                    # weight = Activation('softmax')(self.l2_norm_complex([entangled_real, entangled_image]))
                else:
                    [latent_real, latent_image] = self.dense_1[count]([separable_real, separable_image])
                    # [latent_real, latent_image] = self.complex_l2normilization([latent_real, latent_image])
                    # [latent_real, latent_image] = self.dense_2[n_val - 1]([latent_real, latent_image])
                    [entangled_real, entangled_image] = self.dense_2[count]([latent_real, latent_image])
                    # entangled_real = Add()([entangled_real,separable_real])
                    # entangled_image = Add()([entangled_image, separable_image])

                    [entangled_real, entangled_image] = self.complex_l2normilization([entangled_real, entangled_image])

            probs_list.append(self.measure[count]([entangled_real, entangled_image]))
            count = count +1

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
