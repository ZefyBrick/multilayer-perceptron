#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sys import argv
from argparse import ArgumentParser
import os
import json


class MPTrainer():
    def __init__(self):
        self._parser_trainer()
        self._validation()
        self.INPUT_DIM = 9
        self.OUTPUT_DIM = 2
        self.ALPHA = 10e-5
        self.EPSILON = 10e-6
        self.GAMMA = 0.75
        self.BETA_1 = 0.2
        self.BETA_2 = 0.4
        np.random.seed(10)
        self.loss_err = []
        self.loss_err_predict = []

    def train(self):
        self._init_params()
        self._prepare_data()
        for i in range(self.NUM_EPOCHS):
            """
            Прогоняем параметры по модели, получаем предсказания и ошибку для тестовой и валидационной выборок
            """
            z_predict = self._predict(self.X_test)
            E_predict = self._cross_entropy(z_predict, self.y_test)
            self.loss_err_predict.append(E_predict)
            z = self._predict(self.X_train)
            E = self._cross_entropy(z, self.y_train)
            self.loss_err.append(E)
            if E_predict > self.loss_err_predict[i - 1]:
                self._save_data(i)
                self._create_plot()
            """
            Вычисление параметров точности предсказаний
            """
            accuracy, precision, recall = self._accuracy(z, self.y_train)
            accuracy_predict, precision_predict, recall_predict = self._accuracy(z_predict, self.y_test)
            print(f"""epoch \033[1m\033[31m{i + 1}\033[0m - entropy_loss: \033[1m\033[34m{round(E, 9)}\033[0m - val_entropy_loss: \033[1m\033[34m{round(E_predict, 9)}\033[0m
            accuracy: {round(accuracy, 9)} - val_accuracy: {round(accuracy_predict, 9)}
            precision: {round(precision, 9)} - val_precision: {round(precision_predict, 9)}
            recall: {round(recall, 9)} - val_recall: {round(recall_predict, 9)}""")
            """
            BackPropogation - обратное распространение ошибки
            Начинаем с последнего слоя:
                высчитываем разницу между предсказанными вероятностями и реальными значениями (локальный градиент)
                корректировка весов = локальный градиент * значение функции активации(выход) предыдущего слоя
                корректировка свободного члена = сумма локальных градиентов
            """
            dE_dt4 = z - self.y_train
            self.dE_dW4 = self.h3.T @ dE_dt4
            self.dE_db4 = np.sum(dE_dt4, axis=0)
            """
            Для всех следующих слоёв пользуемся логикой:
                ошибка следующего слоя матрично умноженная на транспонированные веса нынешнего слоя
                локальный градиент = полученное значение * производную ф-ции активации(сигмоида)
                корректировка весов = локальный градиент * значение функции активации(выход) предыдущего слоя
                корректировка свободного члена = сумма локальных градиентов
            """
            dE_dh3 = dE_dt4 @ self.third_layer_weights.T
            dE_dt3 = dE_dh3 * self._deriv_sigmoid(self.t3)
            self.dE_dW3 = self.h2.T @ dE_dt3
            self.dE_db3 = np.sum(dE_dt3, axis=0)
            dE_dh2 = dE_dt3 @ self.second_layer_weights.T
            dE_dt2 = dE_dh2 * self._deriv_sigmoid(self.t2)
            self.dE_dW2 = self.h1.T @ dE_dt2
            self.dE_db2 = np.sum(dE_dt2, axis=0)
            dE_dh1 = dE_dt2 @ self.first_layer_weights.T
            dE_dt1 = dE_dh1 * self._deriv_sigmoid(self.t1)
            self.dE_dW1 = self.X_train.T @ dE_dt1
            self.dE_db1 = np.sum(dE_dt1, axis=0)
            """
            Корректировка весов с помощью одного из алгоритмов оптимизации
            """
            if self.OPTIMIZATION == 'Nesterov':
                self._nesterov()
            elif self.OPTIMIZATION == 'adagrad':
                self._adagrad()
            elif self.OPTIMIZATION == 'rmsprop':
                self._rmsprop()
            elif self.OPTIMIZATION == 'adam':
                self._adam(i)
            else:
                self._gradient()

    def _init_params(self):
        """
        Создаём матрицы весов и биасов для всех слоёв.
        Добавляем в модель дополнительные параметры в зависимости от метода оптимизации
        """
        self.input_weights = np.random.randn(self.INPUT_DIM, self.FIRST_DIM)
        self.input_bias = np.random.randn(self.FIRST_DIM)
        self.first_layer_weights = np.random.randn(self.FIRST_DIM, self.SECOND_DIM)
        self.first_layer_bias = np.random.randn(self.SECOND_DIM)
        self.second_layer_weights = np.random.randn(self.SECOND_DIM, self.THIRD_DIM)
        self.second_layer_bias = np.random.randn(self.THIRD_DIM)
        self.third_layer_weights = np.random.randn(self.THIRD_DIM, self.OUTPUT_DIM)
        self.third_layer_bias = np.random.randn(self.OUTPUT_DIM)
        if self.OPTIMIZATION != 'gradient':
            self.last_opt_input_weights = np.zeros((self.INPUT_DIM, self.FIRST_DIM))
            self.last_opt_input_bias = np.zeros((self.FIRST_DIM))
            self.last_opt_first_layer_weights = np.zeros((self.FIRST_DIM, self.SECOND_DIM))
            self.last_opt_first_layer_bias = np.zeros((self.SECOND_DIM))
            self.last_opt_second_layer_weights = np.zeros((self.SECOND_DIM, self.THIRD_DIM))
            self.last_opt_second_layer_bias = np.zeros((self.THIRD_DIM))
            self.last_opt_third_layer_weights = np.zeros((self.THIRD_DIM, self.OUTPUT_DIM))
            self.last_opt_third_layer_bias = np.zeros((self.OUTPUT_DIM))
            if self.OPTIMIZATION == 'Nesterov':
                self.ALPHA *= 1 - self.GAMMA
            elif self.OPTIMIZATION == 'adam':
                self.ALPHA = 10e-7 * 5
                self.last_opt_v_input_weights = np.zeros((self.INPUT_DIM, self.FIRST_DIM))
                self.last_opt_v_input_bias = np.zeros((self.FIRST_DIM))
                self.last_opt_v_first_layer_weights = np.zeros((self.FIRST_DIM, self.SECOND_DIM))
                self.last_opt_v_first_layer_bias = np.zeros((self.SECOND_DIM))
                self.last_opt_v_second_layer_weights = np.zeros((self.SECOND_DIM, self.THIRD_DIM))
                self.last_opt_v_second_layer_bias = np.zeros((self.THIRD_DIM))
                self.last_opt_v_third_layer_weights = np.zeros((self.THIRD_DIM, self.OUTPUT_DIM))
                self.last_opt_v_third_layer_bias = np.zeros((self.OUTPUT_DIM))
            else:
                self.ALPHA = 10e-7

    def _prepare_data(self):
        """
        Предобрабатываем данные.
        Делим на тестовую и валидационную выборки.
        Делим на фичи и признаки и скалируем данные.
        """
        self.data['diagnoz'] = np.where(self.data[1] == 'M', 1, 0)
        self.data['diagnoz_1'] = np.where(self.data['diagnoz'] == 0, 1, 0)
        self.data = self.data[[8, 13, 16, 20, 23, 25, 26, 30, 31, 'diagnoz', 'diagnoz_1']]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.drop(['diagnoz', 'diagnoz_1'], axis=1),
                                                            self.data[['diagnoz', 'diagnoz_1']],
                                                            test_size=0.2, random_state=21,
                                                            stratify=self.data[['diagnoz', 'diagnoz_1']])
        self.X_train.reset_index(inplace=True, drop=True)
        self.X_test.reset_index(inplace=True, drop=True)
        self.y_train.index = np.arange(self.y_train.shape[0])
        self.y_test.index = np.arange(self.y_test.shape[0])
        self.X_train = self._scale_train()
        self.X_test = self._scale_test(self.X_test)
        self.X_train, self.X_test, self.y_train, self.y_test = np.array(self.X_train), np.array(self.X_test), \
                                                               np.array(self.y_train), np.array(self.y_test)

    def _gradient(self):
        """
        Реализация градиентного спуска.
        w = w - learning_rate * loss_function
        """
        self.input_weights -= self.ALPHA * self.dE_dW1
        self.input_bias -= self.ALPHA * self.dE_db1
        self.first_layer_weights -= self.ALPHA * self.dE_dW2
        self.first_layer_bias -= self.ALPHA * self.dE_db2
        self.second_layer_weights -= self.ALPHA * self.dE_dW3
        self.second_layer_bias -= self.ALPHA * self.dE_db3
        self.third_layer_weights -= self.ALPHA * self.dE_dW4
        self.third_layer_bias -= self.ALPHA * self.dE_db4

    def _nesterov(self):
        """
        Реализация ускоренного градиента Нестерова (Накопление импульсов),
        часть 1
        v(t) = gamma * v(t-1) + (1 - gamma) * learning_rate * loss_function
        """
        self.last_opt_input_weights = self.GAMMA * self.last_opt_input_weights + self.ALPHA * self.dE_dW1
        self.last_opt_input_bias = self.GAMMA * self.last_opt_input_bias + self.ALPHA * self.dE_db1
        self.last_opt_first_layer_weights = self.GAMMA * self.last_opt_first_layer_weights + self.ALPHA * self.dE_dW2
        self.last_opt_first_layer_bias = self.GAMMA * self.last_opt_first_layer_bias + self.ALPHA * self.dE_db2
        self.last_opt_second_layer_weights = self.GAMMA * self.last_opt_second_layer_weights + self.ALPHA * self.dE_dW3
        self.last_opt_second_layer_bias = self.GAMMA * self.last_opt_second_layer_bias + self.ALPHA * self.dE_db3
        self.last_opt_third_layer_weights = self.GAMMA * self.last_opt_third_layer_weights + self.ALPHA * self.dE_dW4
        self.last_opt_third_layer_bias = self.GAMMA * self.last_opt_third_layer_bias + self.ALPHA * self.dE_db4
        """
        Реализация ускоренного градиента Нестерова, часть 2
        w = w - v
        """
        self.input_weights -= self.last_opt_input_weights
        self.input_bias -= self.last_opt_input_bias
        self.first_layer_weights -= self.last_opt_first_layer_weights
        self.first_layer_bias -= self.last_opt_first_layer_bias
        self.second_layer_weights -= self.last_opt_second_layer_weights
        self.second_layer_bias -= self.last_opt_second_layer_bias
        self.third_layer_weights -= self.last_opt_third_layer_weights
        self.third_layer_bias -= self.last_opt_third_layer_bias

    def _rmsprop(self):
        """
        Реализация ускоренного градиента rmsprop
        G(t) = gamma * G(t-1) + (1 - gamma) * loss_function
        """
        self.last_opt_input_weights += (1 - self.GAMMA) * self.dE_dW1**2
        self.last_opt_input_bias += (1 - self.GAMMA) * self.dE_db1**2
        self.last_opt_first_layer_weights += (1 - self.GAMMA) * self.dE_dW2**2
        self.last_opt_first_layer_bias += (1 - self.GAMMA) * self.dE_db2**2
        self.last_opt_second_layer_weights += (1 - self.GAMMA) * self.dE_dW3**2
        self.last_opt_second_layer_bias += (1 - self.GAMMA) * self.dE_db3**2
        self.last_opt_third_layer_weights += (1 - self.GAMMA) * self.dE_dW4**2
        self.last_opt_third_layer_bias += (1 - self.GAMMA) * self.dE_db4**2
        self._new_weights()

    def _adam(self, i):
        """
        Реализация ускоренного градиента Adam
        :param i: номер эпохи
        часть 1
        m(t) = betta_1 * m(t - 1) + (1 - betta_1) * loss_function
        """
        self.last_opt_input_weights = self.BETA_1 * self.last_opt_input_weights + (1 - self.BETA_1) * self.dE_dW1
        self.last_opt_input_bias = self.BETA_1 * self.last_opt_input_bias + (1 - self.BETA_1) * self.dE_db1
        self.last_opt_first_layer_weights = self.BETA_1 * self.last_opt_first_layer_weights + (1 - self.BETA_1) * self.dE_dW2
        self.last_opt_first_layer_bias = self.BETA_1 * self.last_opt_first_layer_bias + (1 - self.BETA_1) * self.dE_db2
        self.last_opt_second_layer_weights = self.BETA_1 * self.last_opt_second_layer_weights + (1 - self.BETA_1) * self.dE_dW3
        self.last_opt_second_layer_bias = self.BETA_1 * self.last_opt_second_layer_bias + (1 - self.BETA_1) * self.dE_db3
        self.last_opt_third_layer_weights = self.BETA_1 * self.last_opt_third_layer_weights + (1 - self.BETA_1) * self.dE_dW4
        self.last_opt_third_layer_bias = self.BETA_1 * self.last_opt_third_layer_bias + (1 - self.BETA_1) * self.dE_db4
        """
        часть 2
        v(t) = betta_2 * v(t - 1) + (1 - betta_2) * loss_function**2
        """
        self.last_opt_v_input_weights = self.BETA_2 * self.last_opt_v_input_weights + (1 - self.BETA_2) * self.dE_dW1**2
        self.last_opt_v_input_bias = self.BETA_2 * self.last_opt_v_input_bias + (1 - self.BETA_2) * self.dE_db1**2
        self.last_opt_v_first_layer_weights = self.BETA_2 * self.last_opt_v_first_layer_weights + (
                    1 - self.BETA_2) * self.dE_dW2**2
        self.last_opt_v_first_layer_bias = self.BETA_2 * self.last_opt_v_first_layer_bias + (1 - self.BETA_2) * self.dE_db2**2
        self.last_opt_v_second_layer_weights = self.BETA_2 * self.last_opt_v_second_layer_weights + (
                    1 - self.BETA_2) * self.dE_dW3**2
        self.last_opt_v_second_layer_bias = self.BETA_2 * self.last_opt_v_second_layer_bias + (
                    1 - self.BETA_2) * self.dE_db3**2
        self.last_opt_v_third_layer_weights = self.BETA_2 * self.last_opt_v_third_layer_weights + (
                    1 - self.BETA_2) * self.dE_dW4**2
        self.last_opt_v_third_layer_bias = self.BETA_2 * self.last_opt_v_third_layer_bias + (1 - self.BETA_2) * self.dE_db4**2
        """
        Корректировка параметров
        m = m / (1 - betta_1**t) if 0 < t < 10_000
        """
        if 0 < i < 1_000:
            self.last_opt_input_weights /= 1 - np.power(self.BETA_1, i)
            self.last_opt_input_bias /= 1 - np.power(self.BETA_1, i)
            self.last_opt_first_layer_weights /= 1 - np.power(self.BETA_1, i)
            self.last_opt_first_layer_bias /= 1 - np.power(self.BETA_1, i)
            self.last_opt_second_layer_weights /= 1 - np.power(self.BETA_1, i)
            self.last_opt_second_layer_bias /= 1 - np.power(self.BETA_1, i)
            self.last_opt_third_layer_weights /= 1 - np.power(self.BETA_1, i)
            self.last_opt_third_layer_bias /= 1 - np.power(self.BETA_1, i)
            """
            Корректировка параметров
            v = v / (1 - betta_2**t) if 0 < t < 10_000
            """
            self.last_opt_v_input_weights /= 1 - np.power(self.BETA_2, i)
            self.last_opt_v_input_bias /= 1 - np.power(self.BETA_2, i)
            self.last_opt_v_first_layer_weights /= 1 - np.power(self.BETA_2, i)
            self.last_opt_v_first_layer_bias /= 1 - np.power(self.BETA_2, i)
            self.last_opt_v_second_layer_weights /= 1 - np.power(self.BETA_2, i)
            self.last_opt_v_second_layer_bias /= 1 - np.power(self.BETA_2, i)
            self.last_opt_v_third_layer_weights /= 1 - np.power(self.BETA_2, i)
            self.last_opt_v_third_layer_bias /= 1 - np.power(self.BETA_2, i)
        """
        Корректировка весов
        w = w - learning_rate * m / (v + epsilon)**0.5
        """
        self.input_weights -= self.ALPHA * self.last_opt_input_weights * (self.last_opt_v_input_weights + self.EPSILON) ** 0.5
        self.input_bias -= self.ALPHA * self.last_opt_input_bias * (self.last_opt_v_input_bias + self.EPSILON) ** 0.5
        self.first_layer_weights -= self.ALPHA * self.last_opt_first_layer_weights * (self.last_opt_v_first_layer_weights + self.EPSILON) ** 0.5
        self.first_layer_bias -= self.ALPHA * self.last_opt_first_layer_bias * (self.last_opt_v_first_layer_bias + self.EPSILON) ** 0.5
        self.second_layer_weights -= self.ALPHA * self.last_opt_second_layer_weights * (
                    self.last_opt_v_second_layer_weights + self.EPSILON) ** 0.5
        self.second_layer_bias -= self.ALPHA * self.last_opt_second_layer_bias * (self.last_opt_v_second_layer_bias + self.EPSILON) ** 0.5
        self.third_layer_weights -= self.ALPHA * self.last_opt_third_layer_weights * (self.last_opt_v_third_layer_weights + self.EPSILON) ** 0.5
        self.third_layer_bias -= self.ALPHA * self.last_opt_third_layer_bias * (self.last_opt_v_third_layer_bias + self.EPSILON) ** 0.5

    def _adagrad(self):
        """
        Реализация ускоренного градиента Adagrad
        G = G + loss_function**2
        w = w - learning_rate * loss_function / (G + epsilon)**0.5
        """
        self.last_opt_input_weights += self.dE_dW1**2
        self.last_opt_input_bias += self.dE_db1**2
        self.last_opt_first_layer_weights += self.dE_dW2**2
        self.last_opt_first_layer_bias += self.dE_db2**2
        self.last_opt_second_layer_weights += self.dE_dW3**2
        self.last_opt_second_layer_bias += self.dE_db3**2
        self.last_opt_third_layer_weights += self.dE_dW4**2
        self.last_opt_third_layer_bias += self.dE_db4**2
        self._new_weights()

    def _new_weights(self):
        """
        вторая часть ускоренных градиентов adagrad & rmsprop
        w = w - learning_rate * loss_function / (G + epsilon)**0.5
        """
        self.input_weights -= self.ALPHA * self.dE_dW1 * (self.last_opt_input_weights + self.EPSILON) ** 0.5
        self.input_bias -= self.ALPHA * self.dE_db1 * (self.last_opt_input_bias + self.EPSILON) ** 0.5
        self.first_layer_weights -= self.ALPHA * self.dE_dW2 * (self.last_opt_first_layer_weights + self.EPSILON) ** 0.5
        self.first_layer_bias -= self.ALPHA * self.dE_db2 * (self.last_opt_first_layer_bias + self.EPSILON) ** 0.5
        self.second_layer_weights -= self.ALPHA * self.dE_dW3 * (self.last_opt_second_layer_weights + self.EPSILON) ** 0.5
        self.second_layer_bias -= self.ALPHA * self.dE_db3 * (self.last_opt_second_layer_bias + self.EPSILON) ** 0.5
        self.third_layer_weights -= self.ALPHA * self.dE_dW4 * (self.last_opt_third_layer_weights + self.EPSILON) ** 0.5
        self.third_layer_bias -= self.ALPHA * self.dE_db4 * (self.last_opt_third_layer_bias + self.EPSILON) ** 0.5

    def _save_data(self, last_epoch):
        weights = np.array(
            [self.input_weights, self.first_layer_weights, self.second_layer_weights, self.third_layer_weights],
            dtype=object)
        np.save('weights.npy', weights)
        biases = np.array([self.input_bias, self.first_layer_bias, self.second_layer_bias, self.third_layer_bias],
                          dtype=object)
        np.save('biases.npy', biases)
        hyperparameters = {'ALPHA': self.ALPHA,
                           'EPSILON': self.EPSILON,
                           'GAMMA': self.GAMMA,
                           'BETA_1': self.BETA_1,
                           'BETA_2': self.BETA_2,
                           'SEED': 10,
                           'NUM_EPOCHS_DEFAULT': self.NUM_EPOCHS,
                           'NUM_EPOCHS': last_epoch,
                           'OPTIMIZATION': self.OPTIMIZATION,
                           'INPUT_DIM': self.INPUT_DIM,
                           'OUTPUT_DIM': self.OUTPUT_DIM,
                           'FIRST_DIM': self.FIRST_DIM,
                           'SECOND_DIM': self.SECOND_DIM,
                           'THIRD_DIM': self.THIRD_DIM
                           }
        with open('hyperparameters.json', 'w') as outfile:
            json.dump(hyperparameters, outfile)

    def _scale_train(self):
        """
        Скалирование данных для обучения и сохранение параметров
        x = (x - meaan) / std
        :return: Обновлённый датасет
        """
        self.scales = [0] * self.X_train.shape[1]
        scale_X = self.X_train.copy()
        for i, k in enumerate(self.X_train.columns):
            std = self.X_train[k].std()
            mean = self.X_train[k].mean()
            self.scales[i] = [std, mean]
            for j in range(self.X_train.shape[0]):
                scale_X[k][j] = (self.X_train[k][j] - mean) / std
        scales = np.array(self.scales,dtype=object)
        np.save('scales.npy', scales)
        return scale_X

    def _scale_test(self, data):
        """
        Скалирование данных для валидации на параметрах обучающей выборки
        x = (x - meaan) / std
        :return: Обновлённый датасет
        """
        scale_X = data.copy()
        for i, k in enumerate(data.columns):
            for j in range(data.shape[0]):
                scale_X[k][j] = (data[k][j] - self.scales[i][1]) / self.scales[i][0]
        return scale_X

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _deriv_sigmoid(self, x):
        # Производная от sigmoid: f'(x) = f(x) * (1 - f(x))
        fx = self._sigmoid(x)
        return fx * (1 - fx)

    def _softmax(self, exit):
        """
        Вероятность принадлежности элемента к каждому из классов
        :param exit: Значения, полученные на последнем слое
        :return: Матрица вероятностей
        """
        out = np.exp(exit)
        exit_layer = out / np.sum(out, axis=1, keepdims=True)
        return exit_layer

    def _cross_entropy(self, z, y):
        """
        Вычисление ошибки классификации
        :param z: Предсказанные значения
        :param y: Реальные значения
        """
        return -np.mean(np.log(z) * y + (1 - y) * np.log(1 - z))

    def _predict(self, input_data):
        """
        Для каждого слоя вычисляем:
            Т = входные параметры или результат вычисления на предыдущем слое * веса + байес
            Н = функция активации от Т
        :param input_data: Входные параметры
        :return: Матрица вероятностей принадлежности к тому или иному классу
        """
        self.t1 = input_data @ self.input_weights + self.input_bias
        self.h1 = self._sigmoid(self.t1)
        self.t2 = self.h1 @ self.first_layer_weights + self.first_layer_bias
        self.h2 = self._sigmoid(self.t2)
        self.t3 = self.h2 @ self.second_layer_weights + self.second_layer_bias
        self.h3 = self._sigmoid(self.t3)
        self.t4 = self.h3 @ self.third_layer_weights + self.third_layer_bias
        return self._softmax(self.t4)

    def _create_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.loss_err, label='Ошибка на обучающей выборке')
        ax.plot(self.loss_err_predict, label='Ошибка на тестовой выборке', color="red")
        ax.legend()
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Ошибка")
        ax.set_title("Изменение значения ошибки кросс-энтропии")
        plt.savefig('error.png')
        exit()

    def _accuracy(self, z, y):
        """
        Расчёт точности модели по 3 метрикам
        accuracy - доля правильных ответов
        precision - доля объектов, названных положительными и при этом являющимися положительными
        recall - доля объектов положительного класса, которые нашёл алгоритм, из всех объектов положительного класса
        :param z: Предсказанные значения
        :param y: Реальные значения
        :return: accuracy, precision, recall
        """
        correct = 0
        precision = 0
        recall = 0
        for i, j in enumerate(z):
            y_pred = np.argmax(j)
            if y_pred != y[i][0]:
                correct += 1
            if y_pred == 0:
                precision += 1
            if y_pred == y[i][0]:
                recall += 1
        accuracy = correct / y.shape[0]
        precision = 1 if precision == 0 else correct / (correct + precision)
        recall = 1 if recall == 0 else correct / (correct + recall)
        return accuracy, precision, recall

    def _parser_trainer(self):
        self.parser = ArgumentParser(prog='train_model', description='''
            Эта программа создаёт модель нейронной сети, обученной на датасете, описывающем параметры злокачественных и доброкачественных клеток''',
                                add_help=True, epilog='''
            (c) April 2022. Автор программы, как всегда, пусечка и лапочка''')
        self.parser.add_argument('--data', '-data', default='data.csv',
                            help='''Датасет для обучения модели''')
        self.parser.add_argument('--optimization', '-optimization',
                                 default='gradient', type=str,
                                 choices=['gradient', 'Nesterov', 'adagrad', 'rmsprop', 'adam'],
                                 help='Количество эпох')
        self.parser.add_argument('--epochs', '-epochs',
                            default=20_000, type=int,
                            choices=range(1, 100_001),
                            help='Количество эпох')
        self.parser.add_argument('--neurons_for_input_layer', '-neurons_for_input_layer',
                                 default=50, type=int,
                                 choices=range(1, 101),
                                 help='Количество нейронов входного слоя')
        self.parser.add_argument('--neurons_for_first_layer', '-neurons_for_first_layer',
                                 default=43, type=int,
                                 choices=range(1, 101),
                                 help='Количество нейронов первого слоя')
        self.parser.add_argument('--neurons_for_second_layer', '-neurons_for_second_layer',
                                 default=31, type=int,
                                 choices=range(1, 101),
                                 help='Количество нейронов второго слоя')

    def _validation(self):
        name = self.parser.parse_args(argv[1:])
        self.data = name.data
        self.NUM_EPOCHS = name.epochs
        self.FIRST_DIM = name.neurons_for_input_layer
        self.SECOND_DIM = name.neurons_for_first_layer
        self.THIRD_DIM = name.neurons_for_second_layer
        self.OPTIMIZATION = name.optimization
        if not os.path.isfile(self.data):
            raise FileNotFoundError('Wrong path for file!')
        try:
            self.data = pd.read_csv(self.data, header=None)
        except Exception as e:
            print(e)
            exit()


if __name__ == '__main__':
    model = MPTrainer()
    model.train()
