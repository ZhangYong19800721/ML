#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""多层感知机"""

__author__ = '张勇,24452861@qq.com'

import numpy as np
from learn.tools.sigmoid import sigmoid
from learn.optimal.minimize_cg import minimize_cg
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self, configure):  # 构造函数
        self.w, self.b = [], []  # 初始化为空列表
        L = len(configure) - 1  # 计算层数
        for l in range(L):
            self.w.append(0.01 * np.random.randn(configure[l + 1], configure[l]))
            self.b.append(np.zeros((configure[l + 1], 1)))

    def __str__(self):
        return "多层感知机"

    __repr__ = __str__

    def applyx(self, x):  # 设置参数x
        i, j = 0, 0
        L = len(self.b)  # 计算层数
        for l in range(L):
            i, j = j, self.w[l].size + j
            self.w[l] = np.reshape(x[i:j], self.w[l].shape)
            i, j = j, self.b[l].size + j
            self.b[l] = np.reshape(x[i:j], self.b[l].shape)

    @property
    def weight(self):  # 提取参数x
        x = np.zeros((0, 1))
        layers = len(self.b)  # 计算层数
        for l in range(layers):
            x = np.vstack((x, np.reshape(self.w[l], (-1, 1)), np.reshape(self.b[l], (-1, 1))))
        return x

    def compute(self, points, i=None):  # 多层感知器的计算过程
        if i is None:
            i = len(self.b)  # 计算层数
        d, n = points.shape
        y = []  # 初始化输出
        for l in range(i):
            y.append(sigmoid(self.w[l].dot(points) + np.tile(self.b[l], (1, n))))
            points = y[l]
        return y

    def gradient(self, x, i=None):  # 计算梯度
        # 初始化
        self.applyx(x)  # 设置权值和偏置值
        D, S, M = self.points.shape  # d数据维度，n样本点数，m样本批数
        L = len(self.b)  # 计算层数

        if i is None:  # 如果没有给出i就计算目标函数在全部训练数据上的梯度
            g = np.zeros(x.shape)  # 初始化梯度
            for i in range(M):
                g += self.gradient(x, i)
            g /= M
            return g
        else:  # 如果给出i就计算目标函数在第i个minibatch上的梯度
            i = i % M
            minibatch, minilabel = self.points[:, :, i], self.labels[:, :, i]  # 取一个minibatch和minilabel
            a = self.compute(minibatch)  # 输入minibatch计算网络各层的输出
            s = []  # 初始化敏感性
            s.insert(0, (a[L - 1] - minilabel).T / D)  # 计算顶层的敏感性
            for l in range(L - 2, -1, -1):  # 反向传播敏感性
                sx, wx, ax = s[0], self.w[l + 1], a[l] * (1 - a[l])
                s.insert(0, sx.dot(wx) * ax.T)

            g = np.zeros((0, 1))  # 初始化梯度
            for l in range(L):
                H, V = self.w[l].shape
                sx = np.tile(np.reshape(s[l].T, (H, 1, S)), (1, V, 1))
                if l == 0:  # 第0层
                    ax = np.tile(np.reshape(minibatch.T, (1, V, S)), (H, 1, 1))
                else:  # 其它层
                    ax = np.tile(np.reshape(a[l - 1].T, (1, V, S)), (H, 1, 1))
                gx = (sx * ax).sum(axis=2) / S
                bx = s[l].sum(axis=0) / S
                g = np.vstack((g, np.reshape(gx, (-1, 1)), np.reshape(bx, (-1, 1))))
            return g

    def ffobject(self, x, i=None):
        # 初始化
        self.applyx(x)  # 设置权值和偏置值
        D, S, M = self.points.shape  # d数据维度，n样本点数，m样本批数
        L = len(self.b)  # 计算层数

        if i is None:  # 如果没有给出i就计算在全部训练数据上的交叉熵
            f = 0
            for i in range(M):
                f += self.ffobject(x, i)
            f /= M
        else:
            i = i % M
            minibatch, minilabel = self.points[:, :, i], self.labels[:, :, i]  # 取一个minibatch和minilabel
            a = self.compute(minibatch)  # 输入minibatch计算网络各层的输出
            a = a[L - 1]
            f = minilabel * np.log(a) + (1 - minilabel) * np.log(1 - a)
            f = -f.mean()
        return f

    def train(self, points, labels, p=None):
        self.points, self.labels = points, labels  # 绑定训练数据
        x0 = self.weight # 设定迭代起始点
        x,y = minimize_cg(self,x0,p) # 共轭梯度下降
        self.applyx(x) # 设定参数
        self.points, self.labels = None, None  # 解除数据绑定


if __name__ == '__main__':
    N = 2000
    x = np.linspace(-2,2,N)
    k = 4
    y = 0.5 + 0.5 * np.sin(k * np.pi * x / 4)
    plt.plot(x,y)
    plt.show()
    
    perceptron = Perceptron((1,10,1))
    print(perceptron)

    x = np.reshape(x,(1,N,1))
    y = np.reshape(y,(1,N,1))
    perceptron.train(x,y)


            

