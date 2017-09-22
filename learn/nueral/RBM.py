#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'约束玻尔兹曼机'

__author__ = '张勇,24452861@qq.com'

import sys
sys.path.append("..")
import numpy as np
from optimal.Derivable import Derivable
from optimal.Objectable import Objectable
from learn.tools import sigmoid
from learn.tools import sample

class RBM(Derivable,Objectable):
    # 构造函数
    def __init__(self, num_visual, num_hidden):
        self.__num_visual = num_visual
        self.__num_hidden = num_hidden
        self.__weight_v2h = np.zeros((self.__num_hidden,self.__num_visual));
        self.__weight_h2v = None;
        self.__visual_bias = np.zeros((self.__num_visual,1));
        self.__hidden_bias = np.zeros((self.__num_hidden,1));
    
    def __str__(self):
        return "约束玻尔兹曼机：[%d,%d]" % (self.__num_visual,self.__num_hidden)
    __repr__ = __str__
    
    # 计算梯度
    def gradient(self,x,i):
        self.__weight_v2h,self.__hidden_bias,self.__visual_bias = x
            
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
        gh = (h_field_0 - h_field_1).dot(np.ones((N,1))) / S
        gv = (minibatch - v_state_1).dot(np.ones((N,1))) / S
            
        weight_cost = 1e-4
        cw = weight_cost * self.__weight_v2h
        return -(gw-cw), -gh, -gv
    
    # 计算重建误差(目标函数)
    def ffobject(self,x,i):
        self.__weight_v2h,self.__hidden_bias,self.__visual_bias  = x # 嵌入参数
        D,S,M = self.points.shape
        i = 1 + i%M
        minibatch = self.points[:,:,i]
        h_bias = self.__hidden_bias.dot(np.ones((1,S)))
        v_bias = self.__visual_bias.dot(np.ones((1,S)))
        
        # 计算重建误差
        h_field_0 = sigmoid(self.__weight_v2h.dot(minibatch) + h_bias)
        h_state_0 = sample(h_field_0)
        v_field_1 = sigmoid(self.__weight_v2h.T.dot(h_state_0) + v_bias)
        y =  ((v_field_1 - minibatch)**2).sum() / S # 计算在整个minibatch上的平均重建误差
        return y
    
    def rebuild(self,x):
        pass
    
    def synchronize(self):
        pass
    
    def pretrain(self):
        pass
    
    def initialize(self):
        pass
    
# 单元测试
if __name__ == '__main__':
    rbm = RBM(1024,500)
    print(rbm)