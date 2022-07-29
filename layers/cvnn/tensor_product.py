import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
import tensorflow as tf
from keras.constraints import unit_norm
from keras.initializers import Ones

class Complex_tensor_product(Layer):

    def __init__(self, ngram =1,**kwargs):
        self.ngram = ngram
        super(Complex_tensor_product, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Complex_tensor_product, self).build(input_shape)  # Be sure to call this somewhere!

    def get_config(self):
        config = {'ngram': self.ngram}
        base_config = super(Complex_tensor_product, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):

        if self.ngram == 1:
            input_real_1 = inputs[0][:,:,0,:]
            input_imag_1 = inputs[1][:,:,0,:]
            self.shape = tuple(K.int_shape(input_real_1))
            return [input_real_1, input_imag_1]

        if self.ngram == 2:
            ndims = len(inputs[0].shape) - 1
            # weight = K.expand_dims(inputs[2])
            # weight = K.repeat_elements(weight, inputs[0].shape[-1], axis = ndims)
            input_real_1 = K.expand_dims(inputs[0][:,:,0,:])  # shape: (None, 60, 10, 1)
            input_imag_1 = K.expand_dims(inputs[1][:,:,0,:])  # shape: (None, 60, 10, 1)
            input_real_2 = K.expand_dims(inputs[0][:,:,1,:])
            input_imag_2 = K.expand_dims(inputs[1][:,:,1,:])
            ###
            # output = (input_real+i*input_imag)(input_real_transpose-i*input_imag_transpose)
            ###

            output_real = K.batch_dot(input_real_2, input_real_1, axes=[ndims, ndims]) \
                          - K.batch_dot(input_imag_2, input_imag_1, axes=[ndims, ndims])
            shape = K.int_shape(output_real)
            output_real = K.reshape(output_real,(-1,shape[1],shape[2]*shape[3]))

            output_imag = K.batch_dot(input_real_2, input_imag_1, axes=[ndims, ndims]) \
                          + K.batch_dot(input_imag_2, input_real_1, axes=[ndims, ndims])
            output_imag = K.reshape(output_imag, (-1, shape[1], shape[2] * shape[3]))
            self.shape = tuple(K.int_shape(output_real))

            return [output_real, output_imag]


        if self.ngram == 3:
            ndims = len(inputs[0].shape) - 1
            input_real_1 = K.expand_dims(inputs[0][:,:,0,:])  # shape: (None, 60, 300, 1)
            input_imag_1 = K.expand_dims(inputs[1][:,:,0,:])  # shape: (None, 60, 300, 1)
            input_real_transpose_2 = K.expand_dims(inputs[0][:,:,1,:], axis=ndims - 1)  # shape: (None, 60, 300)
            input_imag_transpose_2 = K.expand_dims(inputs[1][:,:,1,:], axis=ndims - 1)  # shape: (None, 60, 300)
            ###
            # output = (input_real+i*input_imag)(input_real_transpose-i*input_imag_transpose)
            ###
            output_real = K.batch_dot(input_real_transpose_2, input_real_1, axes=[ndims - 1, ndims]) - K.batch_dot(
                input_imag_transpose_2, input_imag_1, axes=[ndims - 1, ndims])  # shape: (None, 60, 300, 300)
            shape = K.int_shape(output_real)
            output_real = K.reshape(output_real,(-1,shape[1],shape[2]*shape[3]))
            output_imag = K.batch_dot(input_real_transpose_2, input_imag_1, axes=[ndims - 1, ndims]) + K.batch_dot(
                input_imag_transpose_2, input_real_1, axes=[ndims - 1, ndims])  # shape: (None, 60, 300, 300)
            output_imag = K.reshape(output_imag, (-1, shape[1], shape[2] * shape[3]))

            input_real_transpose_3 = K.expand_dims(inputs[0][:,:,2,:], axis=ndims - 1)  # shape: (None, 60, 1, 300)
            input_imag_transpose_3 = K.expand_dims(inputs[1][:,:,2,:], axis=ndims - 1)  # shape: (None, 60, 1, 300)
            #
            input_real = K.expand_dims(output_real)
            input_imag = K.expand_dims(output_imag)
            output_real1 = K.batch_dot(input_real_transpose_3, input_real, axes=[ndims - 1, ndims]) - K.batch_dot(
                input_imag_transpose_3, input_imag, axes=[ndims - 1, ndims])  # shape: (None, 60, 300, 300)
            shape = K.int_shape(output_real1)
            output_real1 = K.reshape(output_real1,(-1,shape[1],shape[2]*shape[3]))
            output_imag1 = K.batch_dot(input_real_transpose_3, input_imag, axes=[ndims - 1, ndims]) + K.batch_dot(
                input_imag_transpose_3, input_real, axes=[ndims - 1, ndims])  # shape: (None, 60, 300, 300)
            output_imag1 = K.reshape(output_imag1, (-1, shape[1], shape[2] * shape[3]))
            self.shape = tuple(K.int_shape(output_real1))
            return [output_real1, output_imag1]


    def compute_output_shape(self, input_shape):
            return [self.shape, self.shape]

class Real_tensor_product(Layer):

    def __init__(self, ngram = 1,**kwargs):
        self.ngram = ngram
        super(Real_tensor_product, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Real_tensor_product, self).build(input_shape)  # Be sure to call this somewhere!

    def get_config(self):
        config = {'ngram': self.ngram}
        base_config = super(Real_tensor_product, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):

        if self.ngram == 1:
            input_real_1 = inputs[:,:,0,:]  # shape: (None, 60, 300, 1)
            self.shape = tuple(K.int_shape(input_real_1))
            return [input_real_1]

        if self.ngram == 2:
            ndims = len(inputs.shape) - 1
            input_real_1 = K.expand_dims(inputs[:,:,0,:])  # shape: (None, 60, 300, 1)
            input_real_transpose_2 = K.expand_dims(inputs[:,:,1,:], axis=ndims - 1)  # shape: (None, 60, 1, 300)
            output_real = K.batch_dot(input_real_transpose_2, input_real_1, axes=[ndims - 1, ndims])  # shape: (None, 60, 300, 300)
            shape = K.int_shape(output_real)
            output_real = K.reshape(output_real,(-1,shape[1],shape[2]*shape[3]))
            self.shape = tuple(K.int_shape(output_real))
            return output_real


        if self.ngram == 3:
            ndims = len(inputs[0].shape) - 1
            input_real_1 = K.expand_dims(inputs[0][:,:,0,:])  # shape: (None, 60, 300, 1)
            input_imag_1 = K.expand_dims(inputs[1][:,:,0,:])  # shape: (None, 60, 300, 1)
            input_real_transpose_2 = K.expand_dims(inputs[0][:,:,1,:], axis=ndims - 1)  # shape: (None, 60, 300)
            input_imag_transpose_2 = K.expand_dims(inputs[1][:,:,1,:], axis=ndims - 1)  # shape: (None, 60, 300)
            ###
            # output = (input_real+i*input_imag)(input_real_transpose-i*input_imag_transpose)
            ###
            output_real = K.batch_dot(input_real_transpose_2, input_real_1, axes=[ndims - 1, ndims]) - K.batch_dot(
                input_imag_transpose_2, input_imag_1, axes=[ndims - 1, ndims])  # shape: (None, 60, 300, 300)
            shape = K.int_shape(output_real)
            output_real = K.reshape(output_real,(-1,shape[1],shape[2]*shape[3]))
            output_imag = K.batch_dot(input_real_transpose_2, input_imag_1, axes=[ndims - 1, ndims]) + K.batch_dot(
                input_imag_transpose_2, input_real_1, axes=[ndims - 1, ndims])  # shape: (None, 60, 300, 300)
            output_imag = K.reshape(output_imag, (-1, shape[1], shape[2] * shape[3]))

            input_real_transpose_3 = K.expand_dims(inputs[0][:,:,2,:], axis=ndims - 1)  # shape: (None, 60, 1, 300)
            input_imag_transpose_3 = K.expand_dims(inputs[1][:,:,2,:], axis=ndims - 1)  # shape: (None, 60, 1, 300)
            #
            input_real = K.expand_dims(output_real)
            input_imag = K.expand_dims(output_imag)
            output_real1 = K.batch_dot(input_real_transpose_3, input_real, axes=[ndims - 1, ndims]) - K.batch_dot(
                input_imag_transpose_3, input_imag, axes=[ndims - 1, ndims])  # shape: (None, 60, 300, 300)
            shape = K.int_shape(output_real1)
            output_real1 = K.reshape(output_real1,(-1,shape[1],shape[2]*shape[3]))
            output_imag1 = K.batch_dot(input_real_transpose_3, input_imag, axes=[ndims - 1, ndims]) + K.batch_dot(
                input_imag_transpose_3, input_real, axes=[ndims - 1, ndims])  # shape: (None, 60, 300, 300)
            output_imag1 = K.reshape(output_imag1, (-1, shape[1], shape[2] * shape[3]))
            self.shape = tuple(K.int_shape(output_real1))
            return [output_real1, output_imag1]


    def compute_output_shape(self, input_shape):
            return self.shape



