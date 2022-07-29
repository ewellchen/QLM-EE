# -*- coding: utf-8 -*-
from models.BasicModel import BasicModel
from keras.layers import Embedding, GlobalMaxPooling1D,Dense, Masking, Flatten,Dropout, Activation,concatenate,Reshape, Permute,Lambda, Subtract
from keras.models import Model, Input, model_from_json, load_model, Sequential
from keras.constraints import unit_norm
from layers import *
import math
import numpy as np

from keras import regularizers
import keras.backend as K
from distutils.util import strtobool
from models import representation as model_factory
from layers import distance

class SiameseNetwork(BasicModel):

    def initialize(self):
        self.question = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.answer = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.neg_answer = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
      
        distances= [distance.get_distance("AESD.AESD",mean="geometric",delta =0.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    distance.get_distance("AESD.AESD",mean="geometric",delta =1,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    distance.get_distance("AESD.AESD",mean="geometric",delta =1.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    distance.get_distance("AESD.AESD",mean="arithmetic",delta =0.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    distance.get_distance("AESD.AESD",mean="arithmetic",delta =1,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    distance.get_distance("AESD.AESD",mean="arithmetic",delta =1.5,c=1,dropout_keep_prob =self.opt.dropout_rate_probs),
                    distance.get_distance("cosine.Cosine",dropout_keep_prob = self.opt.dropout_rate_probs),
                    distance.get_distance("tensor_comb.TensorComb")
                    ]
                    
        self.distance = distances[self.opt.distance_type]

        
#        self.dense = Dense(self.opt.nb_classes, activation=self.opt.activation, kernel_regularizer= regularizers.l2(self.opt.dense_l2))
        # self.dense_last =  Dense(self.opt.nb_classes, activation="softmax")
                
    def __init__(self,opt):
        self.model = model_factory.setup(opt)
        super(SiameseNetwork, self).__init__(opt)
        

    def build(self):

        if self.opt.match_type == 'pairwise':
            q_rep = self.model.get_representation(self.question)
            positive_answer = self.model.get_representation(self.answer)
            negative_answer = self.model.get_representation(self.neg_answer)

            score1 = self.distance([q_rep, positive_answer])
            score2 = self.distance([q_rep, negative_answer])
            basic_loss = MarginLoss(self.opt.margin)([score1,score2])
            
            output=[score1,score2,basic_loss]
            model = Model([self.question, self.answer, self.neg_answer], output)       
        else:
            raise ValueError('wrong input of matching type. Please input pairwise or pointwise.')
        return model
    
