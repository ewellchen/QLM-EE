# -*- coding: utf-8 -*-
from params import Params
from read_data.qa import data_reader
import keras.backend as K
import pandas as pd
from layers.loss import *
from layers.loss.metrics import m_value
from tools.units import to_array, getOptimizer, parse_grid_parameters
import argparse
import itertools
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import os
import random
from models import match as Sia_model
from tools.evaluation import matching_score, write_to_file


def run(params, reader):
    best_metric_dev = 0
    best_metric_test = 0
    test_data = reader.getTest(iterable=False, mode='test')
    dev_data = reader.getTest(iterable=False, mode='dev')
    qdnn = Sia_model.setup(params)
    model = qdnn.getModel()
    model.summary()
    performance = []
    data_generator = None
    if 'onehot' not in params.__dict__:
        params.onehot = 0

    if params.match_type == 'pairwise':
        test_data.append(test_data[0])  # fill a placeholder for the first parameter
        test_data = [to_array(i, reader.max_sequence_length) for i in test_data]
        dev_data.append(dev_data[0])
        dev_data = [to_array(i, reader.max_sequence_length) for i in dev_data]
        model.compile(loss=identity_loss,
                      optimizer=getOptimizer(name=params.optimizer, lr=params.lr),
                      metrics=None,
                      loss_weights=[0.0, 0.0, 1.0])
        data_generator = reader.get_pairwise_samples()

    print('Training the network:')
    for i in range(params.epochs):
        model.fit_generator(data_generator, epochs=1,
                            steps_per_epoch=int(len(reader.datas["train"]["question"].unique()) / reader.batch_size),
                            verbose=True)
        print('Epoch(%d):' % i)
        print('Validation Performance:')
        y_pred = model.predict(x=dev_data, batch_size=32)
        score = matching_score(y_pred, params.onehot, params.match_type)
        dev_metric = reader.evaluate(score, mode="dev")
        print(dev_metric)
        if dev_metric[0] > best_metric_dev:
            model.save("temp/best_3_trec_dev.h5")

        print('Test Performance:')
        y_pred = model.predict(x=test_data, batch_size=32)
        score = matching_score(y_pred, params.onehot, params.match_type)
        test_metric = reader.evaluate(score, mode="test")
        print(test_metric)
        if test_metric[0] > best_metric_test:
            model.save("temp/best_3_trec_test.h5")
        performance.append(dev_metric + test_metric)

    print('Done.')
    return performance


if __name__ == '__main__':

    params = Params()
    parser = argparse.ArgumentParser(description='QLM-EE for Question Answering.')
    parser.add_argument('-gpu_num', dest='gpu_num', help='please enter the gpu number.', default=0)
    parser.add_argument('-config_path', dest='config_path', help='please enter the configuration file path.',
                        default='config/qa_trec.ini')
    parser.add_argument('-grid_search', dest='grid_search', type=bool, help='grid search?', default=False)
    parser.add_argument('-grid_file_path', dest='config_file_path', help='please enter grid search file path.',
                        default='config/grid_parameters(trec-3gram).ini')
    args = parser.parse_args()
    params.parse_config(args.config_path)

    #   Reproducibility Setting
    seed(params.seed)
    set_random_seed(params.seed)
    random.seed(params.seed)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    file_writer = open(params.output_file, 'w')
    if args.grid_search:
        print('Grid Search Begins.')
        grid_parameters = parse_grid_parameters(args.config_file_path)
        print('Grid parameter:')
        for item in grid_parameters.keys():
            print(item, grid_parameters[item])  # 输出
        parameters = [arg for index, arg in enumerate(itertools.product(*grid_parameters.values()))]

        for parameter in parameters:
            params.setup(zip(grid_parameters.keys(), parameter))
            reader = data_reader.DataReader(params)
            performance = run(params, reader)
            write_to_file(file_writer, params.to_string(), performance)
            K.clear_session()

    else:
        reader = data_reader.DataReader(params)
        # for i in range(1):
        performance = run(params, reader)
        write_to_file(file_writer, params.to_string(), performance)
        K.clear_session()

# model.save("temp/best.h5")
