#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'约束玻尔兹曼机'

__author__ = '张勇,24452861@qq.com'

import sys
sys.path.append("..")
import numpy as np
import scipy.io as sio
from optimal.Derivable import Derivable
from optimal.Objectable import Objectable
from optimal.minimize_sgd import minimize_sgd
from tools.sigmoid import sigmoid
from tools.sample import sample

class RBM(Derivable,Objectable):
    # 构造函数
    def __init__(self, num_visual, num_hidden):
        self.__num_visual = num_visual
        self.__num_hidden = num_hidden
        self.__weight_v2h = np.zeros((self.__num_hidden,self.__num_visual))
        self.__weight_h2v = None
        self.__visual_bias = np.zeros((self.__num_visual,1))
        self.__hidden_bias = np.zeros((self.__num_hidden,1))
    
    def __str__(self):
        return "约束玻尔兹曼机：[%d,%d]" % (self.__num_visual,self.__num_hidden)
    __repr__ = __str__
    
    # 计算梯度
    def gradient(self,x,i):
        # 嵌入参数
        V,H,W = self.__num_visual,self.__num_hidden,self.__num_hidden*self.__num_visual
        self.__weight_v2h = x[0:W].reshape((H,V))
        self.__hidden_bias = x[W:(W+H)]
        self.__visual_bias = x[(W+H):]
            
        D,S,M = self.points.shape;
        i = 1 + i%M;
        minibatch = self.points[:,:,i] # 从数据集中取一个minibatch
        h_bias = self.__hidden_bias.dot(np.ones((1,S)));
        v_bias = self.__visual_bias.dot(np.ones((1,S)));
            
        h_field_0 = sigmoid(self.__weight_v2h.dot(minibatch) + h_bias)
        h_state_0 = sample(h_field_0);
        v_field_1 = sigmoid(self.__weight_v2h.T.dot(h_state_0) + v_bias)
        v_state_1 = v_field_1
        h_field_1 = sigmoid(self.__weight_v2h.dot(v_state_1) + h_bias)
        gw = (h_field_0.dot(minibatch.T) - h_field_1.dot(v_state_1.T)) / S
        gh = (h_field_0 - h_field_1).dot(np.ones((S,1))) / S
        gv = (minibatch - v_state_1).dot(np.ones((S,1))) / S
            
        weight_cost = 1e-4
        cw = weight_cost * self.__weight_v2h
        g = np.vstack(((gw-cw).flatten(),gh.flatten(),gv.flatten()))
        return -g
    
    # 计算重建误差(目标函数)
    def ffobject(self,x,i):
        # 嵌入参数
        V,H,W = self.__num_visual,self.__num_hidden,self.__num_hidden*self.__num_visual
        self.__weight_v2h = x[0:W].reshape((H,V))
        self.__hidden_bias = x[W:(W+H)]
        self.__visual_bias = x[(W+H):]
        
        D,S,M = self.points.shape
        i = 1 + i%M
        minibatch = self.points[:,:,i]
        h_bias = self.__hidden_bias.dot(np.ones((1,S)))
        v_bias = self.__visual_bias.dot(np.ones((1,S)))
        
        # 计算重建误差
        h_field_0 = sigmoid(self.__weight_v2h.dot(minibatch) + h_bias)
        h_state_0 = sample(h_field_0)
        v_field_1 = sigmoid(self.__weight_v2h.T.dot(h_state_0) + v_bias)
        y =  ((v_field_1 - minibatch)**2).sum() / (S*D) # 计算在整个minibatch上的平均重建误差
        return y
    
    def rebuild(self,x):
        pass
    
    def synchronize(self):
        pass
    
    def pretrain(self,data):
        # pretrain 对权值进行预训练
        # 使用CD1快速算法对权值进行预训练
        self.points = data # 绑定训练数据
        # 设定迭代起始点
        w = self.__weight_v2h.flatten()
        h = self.__hidden_bias.flatten()
        v = self.__visual_bias.flatten()
        x0 = np.hstack((w,h,v))
        # 随机梯度下降
        x = minimize_sgd(self,x0)
        # 嵌入参数
        V,H,W = self.__num_visual,self.__num_hidden,self.__num_hidden*self.__num_visual
        self.__weight_v2h = x[0:W].reshape((H,V))
        self.__hidden_bias = x[W:(W+H)]
        self.__visual_bias = x[(W+H):]
        # 解除数据绑定
        self.points = None
    
    def initialize(self):
        pass
    
# 单元测试
if __name__ == '__main__':
    mnist_file_name = '../data/mnist.mat'
    mnist = sio.loadmat(mnist_file_name)
    train_images = mnist['mnist_train_images']
    train_labels = mnist['mnist_train_labels']
    train_images = train_images / 255;
    D,N = train_images.shape
    S,M = 100,N//100
    train_images.shape = (D,S,M)
    rbm = RBM(1024,500)
    rbm.pretrain(train_images)
    