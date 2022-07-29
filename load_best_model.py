# -*- coding: utf-8 -*-
import numpy as np
import random
from numpy.random import seed
from tensorflow import set_random_seed
from keras.models import load_model
from params import Params
from read_data.qa import data_reader
from layers import *
from layers.loss import *
from tools.units import to_array
from tools.evaluation import matching_score
from layers.loss.metrics import m_value
import pyqentangle
import heapq

def cal_entanglment(model, layer_aim=21, n_gram=2, word_dim=10, data=[[1]], len_sentence=100):
    e_dim = pow(word_dim, n_gram)
    data_question = data[0]
    data_answer = data[1]
    len_dev = len(data_question)
    get_layer_output = K.function([model.layers[0].input], model.layers[18].get_output_at(0))
    word_q_embeddings = []
    for k in range(len_dev):
        layer_output = get_layer_output([np.reshape(data_question[k].astype(np.float32), [1, len_sentence])])
        word_q_e = np.array(layer_output)
        word_q_embeddings.append(word_q_e)
    word_q_embeddings = np.array(word_q_embeddings)
    word_q_embeddings = np.reshape(word_q_embeddings, [len_dev, 2, len_sentence, e_dim])
    word_q_embeddings_r = np.reshape(word_q_embeddings[:, 0, :, :], (len_dev * len_sentence * e_dim))
    word_q_embeddings_i = np.reshape(word_q_embeddings[:, 1, :, :], (len_dev * len_sentence * e_dim))
    word_q_embeddings_complex = []
    for i in range(len(word_q_embeddings_r)):
        word_q_embeddings_complex.append(complex(word_q_embeddings_r[i], word_q_embeddings_i[i]))
    word_q_embeddings_complex = np.reshape(word_q_embeddings_complex, (len_dev, len_sentence, e_dim))

    entanglement_degree = []
    for i in range(len_dev):
        entangle_degree = []
        for j in range(len_sentence):
            b = np.reshape(word_q_embeddings_complex[i, j], [word_dim, word_dim*word_dim])
            alpha = pyqentangle.schmidt_decomposition(b)
            entangle_d = 0
            for k in range(6):
                entangle_d = entangle_d - pow(alpha[k][0], 2) * np.log(pow(alpha[k][0], 2))
            entangle_degree.append(entangle_d)
        entanglement_degree.append(entangle_degree)
    entanglement_degree = np.array(entanglement_degree)
    entanglement_degree_all = np.reshape(entanglement_degree, [len_dev * len_sentence])
    return entanglement_degree_all


params = Params()
params.parse_config("config/qa_trec.ini")
seed(params.seed)
set_random_seed(params.seed)
random.seed(params.seed)
reader = data_reader.DataReader(params)

model = load_model('temp/best_3_trec_dev.h5',
                   custom_objects={'NGram': NGram, "L2Normalization": L2Normalization, "L2Norm": L2Norm,
                                   "ComplexMeasurement_vector": ComplexMeasurement_vector,
                                   "ComplexMultiply": ComplexMultiply,
                                   "Concatenation": Concatenation, "ComplexDense": ComplexDense,
                                   "Cosine": Cosine, "Complex_tensor_product": Complex_tensor_product,
                                   "MarginLoss": MarginLoss,
                                   "identity_loss": identity_loss, "m_value": m_value,
                                   "complex_L2Normalization": complex_L2Normalization})

# test_data = reader.getTest(iterable=False, mode='test')
# test_data.append(test_data[0])
# test_data = [to_array(i, reader.max_sequence_length) for i in test_data]
# y_pred = model.predict(x=test_data, batch_size=32)
# score = matching_score(y_pred, params.onehot, params.match_type)
# test_metric = reader.evaluate(score, mode="test")
# print(test_metric)

dev_data = reader.getTest(iterable=False, mode='dev')
dev_data.append(dev_data[0])
dev_data = [to_array(i, reader.max_sequence_length) for i in dev_data]
entanglement_degree = cal_entanglment(model, 25, 3, 6, dev_data, 100)
y_pred = model.predict(x=dev_data, batch_size=32)
score = matching_score(y_pred, params.onehot, params.match_type)
dev_metric = reader.evaluate(score, mode="dev")
print(dev_metric)
# entanglement_degree = cal_entanglment(model, 21, 2, 10, dev_data[1], 100)
# max_100 = heapq.nlargest(100,range(len(entanglement_degree_all)), entanglement_degree_all.take)
