# -*- coding: utf-8 -*-

from layers.cvnn.embedding import phase_embedding_layer, amplitude_embedding_layer,amplitude_embedding_layer_mix
from layers.cvnn.multiply import ComplexMultiply
from layers.cvnn.superposition import ComplexSuperposition
from layers.cvnn.dense import ComplexDense
from layers.cvnn.mixture import ComplexMixture
from layers.cvnn.ent_emb import Entangled_embedding
from layers.cvnn.tensor_product import Complex_tensor_product, Real_tensor_product
from layers.cvnn.measurement import ComplexMeasurement_density,ComplexMeasurement_vector,RealMeasurement_vector
from layers.concatenation import Concatenation
from layers.ngram import NGram
from layers.cvnn.utils import GetReal
from layers.l2_norm import L2Norm, L2Norm_complex
from layers.l2_normalization import L2Normalization, complex_L2Normalization
from layers.cvnn.utils import *
from layers.reshape import reshape
from layers.distance.cosine import Cosine
from layers.loss.marginLoss import MarginLoss
from layers.distance import AESD
#def get



