#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""多层感知机"""

__author__ = '张勇,24452861@qq.com'

import numpy as np
from learn.tools.sigmoid import sigmoid
from learn.optimal.minimize_cg import minimize_cg
import matplotlib.pyplot as plt


class Perceptron(object):
    """多层感知机"""

    def __init__(self, configure):
        """构造函数"""
        self.w, self.b = [], []  # 初始化为空列表
        self.points, self.labels = None, None
        layer_num = len(configure) - 1  # 计算层数
        for l in range(layer_num):
            self.w.append(np.zeros((configure[l + 1], configure[l])))
            self.b.append(np.zeros((configure[l + 1], 1)))

    def initialize(self):
        """初始化权值和偏置值"""
        layer_num = len(self.b)  # 计算层数
        for l in range(layer_num):
            self.w[l] = 0.01 * np.random.randn(*self.w[l].shape)
            self.b[l] = np.zeros(self.b[l].shape)

    def __str__(self):
        return "多层感知机"

    __repr__ = __str__

    def apply(self, wb):
        """设置权值和偏置值"""
        i, j = 0, 0
        layer_num = len(self.b)  # 计算层数
        for l in range(layer_num):
            i, j = j, self.w[l].size + j
            self.w[l] = np.reshape(wb[i:j], self.w[l].shape)
            i, j = j, self.b[l].size + j
            self.b[l] = np.reshape(wb[i:j], self.b[l].shape)

    @property
    def weight(self):
        """提取参数x"""
        wb = np.zeros((0, 1))
        layers = len(self.b)  # 计算层数
        for l in range(layers):
            wb = np.vstack((wb, np.reshape(self.w[l], (-1, 1)), np.reshape(self.b[l], (-1, 1))))
        return wb

    def compute(self, points, i=None):
        """多层感知器的计算过程"""
        if i is None:
            i = len(self.b)  # 计算层数
        d, n = points.shape
        output = []  # 初始化输出
        for l in range(i):
            output.append(sigmoid(self.w[l].dot(points) + np.tile(self.b[l], (1, n))))
            points = output[l]
        return output

    def gradient(self, wb, minibatch_index=None):
        """计算梯度"""
        # 初始化
        self.apply(wb)  # 设置权值和偏置值
        dimension, minibatch_size, minibatch_num = self.points.shape  # D数据维度，S样本点数，M样本批数
        layer_num = len(self.b)  # 计算层数

        if minibatch_index is None:  # 如果没有给出i就计算目标函数在全部训练数据上的梯度
            g = np.zeros(wb.shape)  # 初始化梯度
            for minibatch_index in range(minibatch_num):
                g += self.gradient(wb, minibatch_index)
            return g
        else:  # 如果给出minibatch_index就计算目标函数在第这个minibatch上的梯度
            minibatch_index = minibatch_index % minibatch_num
            minibatch = self.points[:, :, minibatch_index]  # 取一个minibatch和minilabel
            minilabel = self.labels[:, :, minibatch_index]  # 取一个minibatch和minilabel
            a = self.compute(minibatch)  # 输入minibatch计算网络各层的输出
            s = []  # 初始化敏感性
            s.insert(0, (a[layer_num - 1] - minilabel).T)  # 计算顶层的敏感性
            for layer_index in range(layer_num - 2, -1, -1):  # 反向传播敏感性
                sx, wx, ax = s[0], self.w[layer_index + 1], a[layer_index] * (1 - a[layer_index])
                s.insert(0, sx.dot(wx) * ax.T)

            g = np.zeros((0, 1))  # 初始化梯度
            for layer_index in range(layer_num):
                hidden_num, visual_num = self.w[layer_index].shape
                sx = np.tile(np.reshape(s[layer_index].T, (hidden_num, 1, minibatch_size)), (1, visual_num, 1))
                if layer_index == 0:  # 第0层
                    ax = np.tile(np.reshape(minibatch.T, (1, visual_num, minibatch_size)), (hidden_num, 1, 1))
                else:  # 其它层
                    ax = np.tile(np.reshape(a[layer_index - 1].T, (1, visual_num, minibatch_size)), (hidden_num, 1, 1))
                gx = (sx * ax).sum(axis=2)
                bx = s[layer_index].sum(axis=0)
                g = np.vstack((g, np.reshape(gx, (-1, 1)), np.reshape(bx, (-1, 1))))
            return g

    def ffobject(self, wb, minibatch_index=None):
        """计算目标函数（交叉熵）"""
        # 初始化
        self.apply(wb)  # 设置权值和偏置值
        dimension, minibatch_size, minibatch_num = self.points.shape  # d数据维度，n样本点数，m样本批数
        layer_num = len(self.b)  # 计算层数

        if minibatch_index is None:  # 如果没有给出i就计算在全部训练数据上的交叉熵
            f = 0
            for minibatch_index in range(minibatch_num):
                f += self.ffobject(wb, minibatch_index)
        else:
            minibatch_index = minibatch_index % minibatch_num
            minibatch = self.points[:, :, minibatch_index]  # 取一个minibatch和minilabel
            minilabel = self.labels[:, :, minibatch_index]  # 取一个minibatch和minilabel
            a = self.compute(minibatch)  # 输入minibatch计算网络各层的输出
            a = a[layer_num - 1]
            f = minilabel * np.log(a) + (1 - minilabel) * np.log(1 - a)
            f = -f.sum()
        return f

    def train(self, points, labels, p=None):
        """训练"""
        self.points, self.labels = points, labels  # 绑定训练数据
        x0 = self.weight  # 设定迭代起始点
        x_optimal, y_optimal = minimize_cg(self, x0, p)  # 共轭梯度下降
        self.apply(x_optimal)  # 设定参数
        self.points, self.labels = None, None  # 解除数据绑定


if __name__ == '__main__':
    np.random.seed(1)
    N = 2000
    x = np.linspace(-2, 2, N)
    k = 4
    y = 0.5 + 0.5 * np.sin(k * np.pi * x / 4)
    plt.plot(x, y)
    plt.show()

    perceptron = Perceptron((1, 10, 1))
    perceptron.initialize()
    print(perceptron)

    x = np.reshape(x, (1, N, 1))
    y = np.reshape(y, (1, N, 1))
    perceptron.train(x, y)
