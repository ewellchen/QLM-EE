import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
import tensorflow as tf
from keras.constraints import unit_norm
from keras.initializers import Ones


class Entangled_embedding(Layer):

    def __init__(self, keep_dims = True, **kwargs):
        # self.output_dim = output_dim
        self.keep_dims = keep_dims
        super(Entangled_embedding, self).__init__(**kwargs)


    def get_config(self):
        config = {}
        base_config = super(Entangled_embedding, self).get_config()
        return dict(list(base_config.items())+list(config.items()))

    def build(self, input_shape):

        super(Entangled_embedding, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        separable_real, separable_image = inputs[0], inputs[1]
        entangled_real, entangled_image = inputs[2], inputs[3]
        sr = K.expand_dims(separable_real, axis=3)
        si = K.expand_dims(separable_image, axis=3)
        er = K.expand_dims(entangled_real, axis=3)
        ei = K.expand_dims(entangled_image, axis=3)
        rr = K.concatenate([sr, er], axis=-1)
        ii = K.concatenate([si, ei], axis=-1)
        output = K.sqrt(0.0000001 + K.sum(rr ** 2, axis=2, keepdims=self.keep_dims) +
                        K.sum(ii ** 2, 2, keepdims=self.keep_dims))
        def soft_max(input,ax = 1):
            x_exp = tf.exp(input)
            partition = tf.reduce_sum(x_exp,axis=ax,keepdims=True)
            return x_exp/partition
        output = soft_max(output, ax=3)
        weight = K.repeat_elements(output, sr.shape[-2], axis=2)
        output_r = sr[:, :, :, 0] * weight[:, :, :, 0] + er[:, :, :, 0] * weight[:, :, :, 1]
        output_i = si[:, :, :, 0] * weight[:, :, :, 0] + ei[:, :, :, 0] * weight[:, :, :, 1]
        return [output_r, output_i]

    def compute_output_shape(self, input_shape):
        # print(type(input_shape[1]))

        return [input_shape[0], input_shape[1]]



if __name__ == '__main__':
    main()
