# -*- coding: utf-8 -*-

import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
import tensorflow as tf



class L2Normalization(Layer):

    def __init__(self, axis = 1, **kwargs):
        # self.output_dim = output_dim
        self.axis = axis
        super(L2Normalization, self).__init__(**kwargs)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(L2Normalization, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.



        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(L2Normalization, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):



        output = K.l2_normalize(inputs,axis = self.axis)
        return output

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))
        
        return input_shape


class complex_L2Normalization(Layer):

    def __init__(self, axis=1, **kwargs):
        # self.output_dim = output_dim
        self.axis = axis
        super(complex_L2Normalization, self).__init__(**kwargs)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(complex_L2Normalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(complex_L2Normalization, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        # sum  = K.sqrt(0.000000001+ K.sum(inputs[0]**2+inputs[1]**2, axis = self.axis, keepdims = True))
        # output_real = tf.div(inputs[0],sum)
        # output_imag = tf.div(inputs[1],sum)
        # self.shape = K.int_shape(output_real)
        input = K.stack(inputs, axis=3)
        output = K.l2_normalize(input, axis=self.axis)
        output_real = output[:, :, :, 0]
        output_imag = output[:, :, :, 1]
        return [output_real,output_imag]

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))

        return input_shape


def main():
    from keras.layers import Input, Dense

    encoding_dim = 50
    input_dim = 300
    a=np.random.random([5,300])
    input_img = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim)(input_img) #,
    new_code=L2Normalization()(encoded)

    encoder = Model(input_img, new_code)

    b=encoder.predict(a)
    print(np.linalg.norm(b,axis=1))



    # # y = K.sum(K.square(x), axis=None, keepdims = False)
    # x = x/np.linalg.norm(x, ord = 2, axis = (1,2))
    # # print(np.linalg.norm(x[0], ord = 2))
    #     # # print(np.linalg.norm(x))
    # y = model.predict(x)
    # model.fit(x,y)
    # for i in range(100):
    #     x = np.random.random((1,10,2))
    #     x = x/np.linalg.norm(x, ord = 2, axis = (1,2))
    #     y = model.predict(x)
    #     print(y)


if __name__ == '__main__':
    main()
