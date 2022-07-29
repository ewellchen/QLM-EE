import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
from keras.initializers import RandomUniform,RandomNormal
from keras.constraints import unit_norm

import math
from keras.layers import Embedding
from keras import regularizers

def phase_embedding_layer(input_shape, input_dim, embedding_dim = 1,trainable = True,l2_reg=0.0000005, seed = None):
    embedding_layer = Embedding(input_dim,
                            embedding_dim,
                            embeddings_initializer=RandomUniform(minval=0, maxval=2*math.pi, seed=seed),
                            #embeddings_initializer=RandomNormal(mean=0.0, stddev=0.1, seed=None),
                            input_length=input_shape, trainable = trainable,
                            embeddings_regularizer= regularizers.l2(l2_reg))
    return embedding_layer



def amplitude_embedding_layer_mix(embedding_matrix, input_shape, embedding_dim = 1, trainable = False, random_init = False,l2_reg=0.0000005):
    # embedding_dim = embedding_matrix.shape[0]
    vocabulary_size = embedding_matrix.shape[1]
    if(random_init):
        return(Embedding(vocabulary_size,
                                embedding_dim,
                                embeddings_constraint = unit_norm(axis = 1),
                                input_length=input_shape,embeddings_regularizer= regularizers.l2(l2_reg),
                                trainable=trainable))
    else:
        return(Embedding(vocabulary_size,
                                embedding_dim,
                                weights=[np.transpose(embedding_matrix)],
                                embeddings_constraint = unit_norm(axis = 1),
                                input_length=input_shape,embeddings_regularizer= regularizers.l2(l2_reg),
                                trainable=trainable))

def amplitude_embedding_layer(input_shape, input_dim, embedding_dim = 1,trainable = False, l2_reg=0.0000005,seed = None):
    return(Embedding(input_dim,
                            embedding_dim,
                            # embeddings_initializer=RandomUniform(minval=0, maxval=1),
                            # embeddings_initializer=RandomNormal(mean=0.0, stddev=0.1, seed=seed),
                            embeddings_initializer= 'he_normal',
                            embeddings_constraint = unit_norm(axis = 1),
                            input_length=input_shape,
                            embeddings_regularizer= regularizers.l2(l2_reg),
                            trainable=trainable))


