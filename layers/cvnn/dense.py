#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Chiheb Trabelsi
#

import tensorflow as tf
from keras import backend as K
import sys; sys.path.append('.')
from keras import activations, initializers, regularizers, constraints
from keras.layers import Layer, InputSpec
import numpy as np
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class ComplexDense(Layer):

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 init_criterion='glorot',
                 kernel_initializer='complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ComplexDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.init_criterion = init_criterion
        # if kernel_initializer in {'complex'}:
        #     self.kernel_initializer = kernel_initializer
        # else:
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed
        # self.input_spec = InputSpec(ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        # assert len(input_shape) == 2
        # assert input_shape[-1] % 2 == 0
        # input_dim = input_shape[-1] // 2
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(input_shape) != 2:
             raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                              'Got ' + str(len(input_shape)) + ' inputs.')

        input_dim = input_shape[0][-1]
        data_format = K.image_data_format()
        kernel_shape = (input_dim, self.units)
        fan_in, fan_out = initializers._compute_fans(
            kernel_shape,
            data_format=data_format
        )
        fan_in = tf.to_float(fan_in)
        fan_out = tf.to_float(fan_out)
        
        if self.init_criterion == 'he':
            s = K.sqrt(1. / fan_in)
        elif self.init_criterion == 'glorot':
            s = K.sqrt(1. / (fan_in + fan_out))

        # rng = RandomStreams(seed=self.seed)

        # Equivalent initialization using amplitude phase representation:
        """modulus = rng.rayleigh(scale=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        def init_w_real(shape, dtype=None):
            return modulus * K.cos(phase)
        def init_w_imag(shape, dtype=None):
            return modulus * K.sin(phase)"""

        # Initialization using euclidean representation:
        def init_w_real(shape, dtype=None):
            
            return tf.random_normal(
                shape = kernel_shape,
                mean=0.5,
                stddev=s,
                dtype=tf.float32,
                seed=self.seed,
                name=None
            )
            # return rng.normal(
            #     size=kernel_shape,
            #     avg=0,
            #     std=s,
            #     dtype=dtype
            # )
      
        def init_w_imag(shape, dtype=None):
            return tf.random_normal(
                shape = kernel_shape,
                mean=0.5,
                stddev=s,
                dtype=tf.float32,
                seed=self.seed,
                name=None
            )
            # return rng.normal(
            #     size=kernel_shape,
            #     avg=0,
            #     std=s,
            #     dtype=dtype
            # )
        if self.kernel_initializer in {'complex'}:
            real_init = init_w_real
            imag_init = init_w_imag
        else:
            real_init = self.kernel_initializer
            imag_init = self.kernel_initializer

        self.real_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=real_init,
            name='real_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self.imag_kernel = self.add_weight(
            shape=kernel_shape,
            initializer=imag_init,
            name='imag_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(2 * self.units,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None

        # self.input_spec = InputSpec(ndim=2, axes={-1: 2 * input_dim})
        self.built = True

    def call(self, inputs):
        # input_shape = K.shape(inputs)
        # input_dim = input_shape[-1] // 2
        # real_input = inputs[:, :input_dim]
        # imag_input = inputs[:, input_dim:]
        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')
        real_input = inputs[0]
        imag_input = inputs[1]

        inputs = K.concatenate([real_input, imag_input], axis = -1)

        # print(inputs.shape)
        # print(self.real_kernel.shape)
        # print(self.imag_kernel.shape)
        cat_kernels_4_real = K.concatenate(
            [self.real_kernel, -self.imag_kernel],
            axis=-1
        )

        cat_kernels_4_imag = K.concatenate(
            [self.imag_kernel, self.real_kernel],
            axis=-1
        )
        cat_kernels_4_complex = K.concatenate(
            [cat_kernels_4_real, cat_kernels_4_imag],
            axis=0
        )

        output = K.dot(inputs, cat_kernels_4_complex)
        # print(output.shape)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        output_r = output[:,:,0:self.units]
        output_i = output[:, :, self.units: 2*self.units]

        return [output_r,output_i]

    def compute_output_shape(self, input_shape):
        # assert input_shape[-1]
        # output_shape = list(input_shape)
        # output_shape[-1] = 2 * self.units
        output_shape = list(input_shape[0])
        output_shape[-1] = self.units
        return [output_shape, output_shape]

    def get_config(self):
        if self.kernel_initializer in {'complex'}:
            ki = self.kernel_initializer
        else:
            ki = initializers.serialize(self.kernel_initializer)
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'init_criterion': self.init_criterion,
            'kernel_initializer': ki,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'seed': self.seed,
        }
        base_config = super(ComplexDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

