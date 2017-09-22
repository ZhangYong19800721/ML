#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""约束玻尔兹曼机"""

__author__ = '张勇,24452861@qq.com'

import numpy as np
import scipy.io as sio
from learn.optimal.Derivable import Derivable
from learn.optimal.Objectable import Objectable
from learn.optimal.minimize_sgd import minimize_sgd
from learn.tools.sigmoid import sigmoid
from learn.tools.sample import sample


class RBM(Derivable, Objectable):
    # 构造函数
    def __init__(self, num_visual, num_hidden):
        self.__num_visual = num_visual
        self.__num_hidden = num_hidden
        self.__weight_v2h = np.zeros((self.__num_hidden, self.__num_visual))
        self.__weight_h2v = None
        self.__visual_bias = np.zeros((self.__num_visual, 1))
        self.__hidden_bias = np.zeros((self.__num_hidden, 1))

    def __str__(self):
        return "约束玻尔兹曼机：[%d,%d]" % (self.__num_visual, self.__num_hidden)

    __repr__ = __str__

    # 计算梯度
    def gradient(self, x, i):
        # 嵌入参数
        v, h, w = self.__num_visual, self.__num_hidden, self.__num_hidden * self.__num_visual
        self.__weight_v2h = x[0:w].reshape((h, v))
        self.__hidden_bias = x[w:(w + h)]
        self.__visual_bias = x[(w + h):]

        d, s, m = self.points.shape
        i = i % m
        minibatch = self.points[:, :, i]  # 从数据集中取一个minibatch
        h_bias = np.tile(self.__hidden_bias, (1, s))
        v_bias = np.tile(self.__visual_bias, (1, s))

        h_field_0 = sigmoid(self.__weight_v2h.dot(minibatch) + h_bias)
        h_state_0 = sample(h_field_0)
        v_field_1 = sigmoid(self.__weight_v2h.T.dot(h_state_0) + v_bias)
        v_state_1 = v_field_1
        h_field_1 = sigmoid(self.__weight_v2h.dot(v_state_1) + h_bias)
        gw = (h_field_0.dot(minibatch.T) - h_field_1.dot(v_state_1.T)) / s
        gh = (h_field_0 - h_field_1).dot(np.ones((s, 1))) / s
        gv = (minibatch - v_state_1).dot(np.ones((s, 1))) / s

        weight_cost = 1e-4
        cw = weight_cost * self.__weight_v2h
        g = np.vstack(((gw - cw).reshape((-1, 1)), gh, gv))
        return -g

    # 计算重建误差(目标函数)
    def ffobject(self, x, i):
        # 嵌入参数
        v, h, w = self.__num_visual, self.__num_hidden, self.__num_hidden * self.__num_visual
        self.__weight_v2h = x[0:w].reshape((h, v))
        self.__hidden_bias = x[w:(w + h)]
        self.__visual_bias = x[(w + h):]

        d, s, m = self.points.shape
        i = i % m
        minibatch = self.points[:, :, i]
        h_bias = self.__hidden_bias.dot(np.ones((1, s)))
        v_bias = self.__visual_bias.dot(np.ones((1, s)))

        # 计算重建误差
        h_field_0 = sigmoid(self.__weight_v2h.dot(minibatch) + h_bias)
        h_state_0 = sample(h_field_0)
        v_field_1 = sigmoid(self.__weight_v2h.T.dot(h_state_0) + v_bias)
        y = ((v_field_1 - minibatch) ** 2).sum() / (d*s)  # 计算在整个minibatch上的平均重建误差
        return y

    def rebuild(self, x):
        y = self.posterior_sample(x)
        z = self.likelihood(y)
        return z

    def posterior_sample(self, v_state):
        # posterior_sample 计算后验概率采样
        # 在给定显层神经元取值的情况下，对隐神经元进行抽样
        h_state = sample(self.posterior(v_state))
        return h_state

    def posterior(self, v_state):
        # posterior 计算后验概率
        # 在给定显层神经元取值的情况下，计算隐神经元的激活概率
        h_field = sigmoid(self.foreward(v_state))
        return h_field

    def likelihood_sample(self, h_state):
        # likelihood_sample 计算似然概率采样
        # 在给定隐层神经元取值的情况下，对显神经元进行抽样
        v_state = sample(self.likelihood(h_state))
        return v_state

    def likelihood(self, h_state):
        # likelihood 计算似然概率
        # 在给定隐层神经元取值的情况下，计算显神经元的激活概率
        v_field = sigmoid(self.backward(h_state))
        return v_field

    def foreward(self, x):
        y = self.__weight_v2h.dot(x) + np.tile(self.__hidden_bias, (1, x.shape[1]))
        return y

    def backward(self, y):
        if self.__weight_h2v is None:
            x = self.__weight_v2h.T.dot(y) + np.tile(self.__visual_bias, (1, y.shape[1]))
        else:
            x = self.__weight_h2v.T.dot(y) + np.tile(self.__visual_bias, (1, y.shape[1]))
        return x

    def synchronize(self):
        self.__weight_h2v = self.__weight_v2h

    def pretrain(self, data, p=None):
        # pretrain 对权值进行预训练
        # 使用CD1快速算法对权值进行预训练

        self.initialize(data)  # 初始化权值
        self.points = data  # 绑定训练数据
        # 设定迭代起始点
        wt, hb, vb = self.__weight_v2h.reshape((-1, 1)), self.__hidden_bias, self.__visual_bias
        x0 = np.vstack((wt, hb, vb))
        # 随机梯度下降
        x, y = minimize_sgd(self, x0, p)
        # 嵌入参数
        v, h, w = self.__num_visual, self.__num_hidden, self.__num_hidden * self.__num_visual
        self.__weight_v2h = x[0:w].reshape((h, v))
        self.__hidden_bias = x[w:(w + h)]
        self.__visual_bias = x[(w + h):]
        # 解除数据绑定
        self.points = None

    def initialize(self, data):
        # INTIALIZE 初始化权值矩阵为0附近的小随机数，初始化显层神经元的偏置为先验概率，初始化隐层神经元的偏置为0.
        self.__weight_v2h = 0.01 * np.random.random(self.__weight_v2h.shape)
        d, s, m = data.shape
        data.shape = (d, -1)
        self.__visual_bias = np.reshape(np.mean(data, axis=1), (d, 1))  #
        self.__visual_bias = np.log(self.__visual_bias / (1 - self.__visual_bias))
        self.__visual_bias[self.__visual_bias < -100] = -100
        self.__visual_bias[self.__visual_bias > +100] = +100
        self.__hidden_bias = np.zeros(self.__hidden_bias.shape)
        data.shape = (d, s, m)


# 单元测试
if __name__ == '__main__':
    mnist_file_name = '../data/mnist.mat'
    mnist = sio.loadmat(mnist_file_name)
    train_images = mnist['mnist_train_images']
    train_labels = mnist['mnist_train_labels']
    train_images = train_images / 255
    D, N = train_images.shape
    S, M = 100, N // 100
    train_images.shape = (D, S, M)
    rbm = RBM(784, 500)
    parameters = {'decay': 100, 'max_it': M * 100}
    rbm.pretrain(train_images, parameters)
    train_images.shape = (D, -1)
    rebuild_images = rbm.rebuild(train_images)
    error = np.sum((rebuild_images - train_images) ** 2) / (D*N)
    print('总体重建误差:%f' % error)
