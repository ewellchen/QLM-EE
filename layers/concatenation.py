# -*- coding: utf-8 -*-

import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
import sys

from copy import copy

class Concatenation(Layer):

    def __init__(self, axis = 1, **kwargs):
        # self.output_dim = output_dim
        self.axis = axis
        super(Concatenation, self).__init__(**kwargs)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Concatenation, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.



        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(Concatenation, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):



        output = K.concatenate(inputs,axis = self.axis)
        return output

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))

        if self.axis<0:
            self.axis = self.axis + len(input_shape[0])
        new_dim = sum([single_shape[self.axis]  for single_shape in input_shape])
        output_shape = list(input_shape[0])
        output_shape[self.axis] = new_dim

#        print('Input shape concatenation layer:{}'.format(input_shape))
#        print([output_shape])
        return [tuple(output_shape)]


