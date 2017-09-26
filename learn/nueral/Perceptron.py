#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""多层感知机"""

__author__ = '张勇,24452861@qq.com'

import numpy as np


class Perceptron(object):
    def __init__(self,configure): # 构造函数
        self.w,self.b = [],[] # 初始化为空列表
        layers = len(configure) - 1 # 计算层数
        for l in range(layers):
            self.w[l].append(np.zeros((configure(l+1),configure(l))))
            self.b[l].append(np.zeros((configure(l+1),1)))

    def apply(self,x): # 设置参数x
        i,j = 0,0
        layers = len(self.b) # 计算层数
        for l in range(layers):
            i,j = j,self.w[l].size
            self.w[l] = np.reshape(x[i:j],self.w[l].shape)
            i,j = j,self.b[l].size
            self.b[l] = np.reshape(x[i:j],self.b[l].shape)

    @property
    def wxb(self): # 提取参数x
        x = np.zeros((0,1))
        layers = len(self.b) # 计算层数
        for l in range(layers):
            x = np.vstack((x,self.w[l].flatten(),self.b[l].flatten()))
        return x

    def compute(self, points, i=None): # 多层感知器的计算过程
		if i is None:
			i = len(self.b) # 计算层数
		d,n = points.shape
		y = [] # 初始化输出
        for l in range(i)
            y.append(sigmoid(self.w[l] * points + np.tile(self.b[l],1,n)))
            points = y[l]
        return y

    def gradient(self, x, i=None): # 计算梯度
        # 初始化
        self.apply(x) # 设置权值和偏置值
        d, n, m = self.points.shape # d数据维度，n样本点数，m样本批数

        if i is None: # 如果没有给出i就计算目标函数在全部训练数据上的梯度
            g = np.zeros(x.shape) # 初始化梯度
            for i in range(m):
                g += self.gradient(x,i)
            g /= n*m
            return g
        else: # 如果给出i就计算目标函数在第i个minibatch上的梯度
            i = i % m
            minibatch,minilabel = self.points[:,:,i],self.labels[:,:,i] # 取一个minibatch和minilabel
            g = np.zeros(x.shape) # 初始化梯度
            layers = len(self.b)  # 计算层数
            o = self.compute(minibatch) # 输入minibatch计算网络各层的输出
            s = [] # 初始化敏感性
            s.insert(0,((minilabel - o[layers-1]) * o[layers-1] * (1 - o[layers-1])).T / n) # 计算顶层的敏感性
            for l in range(layers-1,0,-1) # 反向传播敏感性
                sx = s{m + 1};
                wx = w{m + 1};
                ax = a{m}. * (1 - a{m});
                sm = zeros(N, obj.num_hidden{m});
        pass

    def ffobject(self, x, i=None):
        pass

    def train(self,points,labels,p=None):
        self.points,self.labels = points,labels  # 绑定训练数据

        self.points,self.labels = None,None # 解除数据绑定


if __name__ == '__main__':
    pass
