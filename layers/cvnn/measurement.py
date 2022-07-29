import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
from keras.constraints import unit_norm
from keras.initializers import Orthogonal
import tensorflow as tf

class ComplexMeasurement_density(Layer):

    def __init__(self, units = 5, trainable = True,  **kwargs):

        self.units = units
        self.trainable = trainable
        self.measurement_constrain = unit_norm(axis = (1,2))
        self.measurement_initalizer = Orthogonal(gain=1.0, seed=666)
        super(ComplexMeasurement_density, self).__init__(**kwargs)

    def get_config(self):
        config = {'units': self.units, 'trainable': self.trainable}
        base_config = super(ComplexMeasurement_density, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(input_shape) != 2:
             raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                              'Got ' + str(len(input_shape)) + ' inputs.')
        self.dim = input_shape[0][-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.units, self.dim,2),
                                      constraint = self.measurement_constrain,
                                      initializer= self.measurement_initalizer,
                                      trainable=self.trainable)
        super(ComplexMeasurement_density, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                            'on a list of 2 inputs.'
                            'Got ' + str(len(inputs)) + ' inputs.')


        kernel_real = self.kernel[:,:,0]
        kernel_imag = self.kernel[:,:,1]

        input_real = inputs[0]
        input_imag = inputs[1]
        # print(input_real.shape)
        # print(input_imag.shape)
        kernel_r = K.batch_dot(K.expand_dims(kernel_real,1), K.expand_dims(kernel_real,2), axes = (1,2)) - K.batch_dot(K.expand_dims(kernel_imag,1), K.expand_dims(kernel_imag,2), axes = (1,2))

        kernel_i = K.batch_dot(K.expand_dims(kernel_imag,1), K.expand_dims(kernel_real,2), axes = (1,2)) + K.batch_dot(K.expand_dims(kernel_real,1), K.expand_dims(kernel_imag,2), axes = (1,2))

        kernel_r = K.reshape(kernel_r, shape = (self.units, self.dim * self.dim))
        kernel_i = K.reshape(kernel_i, shape = (self.units, self.dim * self.dim))


        new_shape = [-1]
        for i in input_real.shape[1:-2]:
            new_shape.append(int(i))
            
        new_shape.append(self.dim*self.dim)
        input_real = K.reshape(input_real, shape = tuple(new_shape))
        input_imag = K.reshape(input_imag, shape = tuple(new_shape))

        output = K.dot(input_real,K.transpose(kernel_r)) - K.dot(input_imag,K.transpose(kernel_i))


        return(output)



    def compute_output_shape(self, input_shape):
        output_shape = [None]
        for i in input_shape[0][1:-2]:
            output_shape.append(i)
        output_shape.append(self.units)
#        output_shape = [input_shape[0][0:-3],self.units]
        
#        print('Input shape of measurment layer:{}'.format(input_shape))
#        print(output_shape)
        return[tuple(output_shape)]


class ComplexMeasurement_vector(Layer):

    def __init__(self,units=5, trainable=True, **kwargs):
        self.units = units
        self.trainable = trainable
        super(ComplexMeasurement_vector, self).__init__(**kwargs)
        self.measurement_constrain = unit_norm(axis=(1, 2))
        self.measurement_initalizer = Orthogonal(gain=1.0, seed=None)

    def get_config(self):
        config = {'units': self.units, 'trainable': self.trainable}
        base_config = super(ComplexMeasurement_vector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.units, self.dim, 2),
                                      constraint = self.measurement_constrain,
                                      initializer= self.measurement_initalizer,
                                      trainable=self.trainable)

        super(ComplexMeasurement_vector, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        # if len(inputs) != 2:
        #     raise ValueError('This layer should be called '
        #                      'on a list of 2 inputs.'
        #                      'Got ' + str(len(inputs)) + ' inputs.')
        # weight = K.expand_dims(inputs[2])
        # weight = K.repeat_elements(weight, pow(self.dim, n_val), axis=3)
        # weight_new = 1
        # for i in range(K.int_shape(inputs[2])[-1]):
        #     weight_new = weight_new * weight[:, :, i]



        kernel_real = self.kernel[:, :, 0] #(500, 64)
        kernel_imag = self.kernel[:, :, 1] #(500, 64)
        self.shape = K.int_shape(inputs[0])
        input_real = inputs[0] #(None, 42,64)
        input_imag = inputs[1]
        # print(input_real.shape)
        # print(input_imag.shape)

        result_real = K.dot(input_real, K.transpose(kernel_real)) - K.dot(input_imag, K.transpose(kernel_imag))
        result_imag = K.dot(input_imag, K.transpose(kernel_real)) + K.dot(input_real, K.transpose(kernel_imag))

        output = K.pow(result_real,2) + K.pow(result_imag,2)

        self.shape = K.int_shape(output)

        return (output)

    def compute_output_shape(self, input_shape):

        return self.shape

class RealMeasurement_vector(Layer):

    def __init__(self,units=5, trainable=True, **kwargs):
        self.units = units
        self.trainable = trainable
        super(RealMeasurement_vector, self).__init__(**kwargs)
        self.measurement_constrain = unit_norm(axis=(1, 2))
        self.measurement_initalizer = Orthogonal(gain=1.0, seed=None)

    def get_config(self):
        config = {'units': self.units, 'trainable': self.trainable}
        base_config = super(RealMeasurement_vector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):

        self.dim = input_shape[-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.units, self.dim,1),
                                      constraint = self.measurement_constrain,
                                      initializer= self.measurement_initalizer,
                                      trainable=self.trainable)

        super(RealMeasurement_vector, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        kernel_real = self.kernel[:, :, 0] #(500, 64)
        self.shape = K.int_shape(inputs)
        input_real = inputs #(None, 42,64)
        # print(input_real.shape)
        # print(input_imag.shape)

        result_real = K.dot(input_real, K.transpose(kernel_real))

        output = K.pow(result_real,2)

        self.shape = K.int_shape(output)

        return (output)

    def compute_output_shape(self, input_shape):

        return self.shape

