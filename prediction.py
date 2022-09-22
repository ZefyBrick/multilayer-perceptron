#!/usr/bin/env python
# -*- coding: utf-8 -*-

from train_model import MPTrainer
import numpy as np
import pandas as pd
from sys import argv
from argparse import ArgumentParser
import os


class Prediction(MPTrainer):
    def __init__(self):
        self._parser_trainer()
        self._validation()

    def prediction(self):
        self._init_params()
        self.scales = np.load('scales.npy', allow_pickle=True)
        self._prepare_data()
        z = self._predict(self.X)
        E = self._cross_entropy(z, self.y)
        accuracy, precision, recall = self._accuracy(z, self.y)
        print(
            f"""entropy_loss: \033[1m\033[34m{round(E, 9)}\033[0m
    accuracy: {round(accuracy, 9)}
    precision: {round(precision, 9)}
    recall: {round(recall, 9)}""")
        np.savetxt("predict.csv", z, delimiter=",", fmt='%1.6f')

    def _prepare_data(self):
        self.data['diagnoz'] = np.where(self.data[1] == 'M', 1, 0)
        self.data['diagnoz_1'] = np.where(self.data['diagnoz'] == 0, 1, 0)
        self.data = self.data[[8, 13, 16, 20, 23, 25, 26, 30, 31, 'diagnoz', 'diagnoz_1']]
        self.y = self.data[['diagnoz', 'diagnoz_1']]
        self.X = self.data.drop(['diagnoz', 'diagnoz_1'], axis=1)
        self.X = self._scale_test(self.X)
        self.X, self.y = np.array(self.X), np.array(self.y)

    def _init_params(self):
        weights = np.load('weights.npy', allow_pickle=True)
        biases = np.load('biases.npy', allow_pickle=True)
        self.input_weights = weights[0]
        self.input_bias = biases[0]
        self.first_layer_weights = weights[1]
        self.first_layer_bias = biases[1]
        self.second_layer_weights = weights[2]
        self.second_layer_bias = biases[2]
        self.third_layer_weights = weights[3]
        self.third_layer_bias = biases[3]

    def _parser_trainer(self):
        self.parser = ArgumentParser(prog='prediction', description='''
            Предсказание злокачественности клетки на основе обученной нейронной сети''',
                                add_help=True, epilog='''
            (c) April 2022. Автор программы, как всегда, пусечка и лапочка''')
        self.parser.add_argument('--data', '-data', default='data_test.csv',
                                 help='''Датасет для обучения модели''')

    def _validation(self):
        name = self.parser.parse_args(argv[1:])
        self.data = name.data
        try:
            self.data = pd.read_csv(self.data, header=None)
        except Exception as e:
            print('Wrong path for file!')
            exit()


if __name__ == '__main__':
    prediction = Prediction()
    prediction.prediction()
