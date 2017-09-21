#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'约束玻尔兹曼机'

__author__ = '张勇,24452861@qq.com'

import sys
sys.path.append("..")

import numpy as np
from optimal.Derivable import Derivable
from optimal.Objectable import Objectable
#import learn.tools

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
    def gradient(self,x,i=None):
        H = self.__num_hidden
        V = self.__num_visual
        W = H*V
        
        self.__weight_v2h = x[0:W];
        #self.hidden_bias = x[W+[0:H]];
#        self.visual_bias = x[W+H+[1:V]];
#            
#        [D,S,M] = self.points.shape;
#        i = 1 + mod(i,M);
#        minibatch = self.points(:,:,i);
#        N = size(minibatch,2); # 训练样本的个数
#        h_bias = repmat(self.hidden_bias,1,N);
#        v_bias = repmat(self.visual_bias,1,N);
#            
#        
#        h_field_0 = tools.sigmoid(self.weight_v2h   * minibatch + h_bias);
#        h_state_0 = tools.sample(h_field_0);
#        v_field_1 = tools.sigmoid(self.weight_v2h.T * h_state_0 + v_bias);
#        v_state_1 = v_field_1; 
#        h_field_1 = tools.sigmoid(self.weight_v2h * v_state_1 + h_bias);
#        gw = (h_field_0 * minibatch.T - h_field_1 * v_state_1.T) / N;
#        gh = (h_field_0 - h_field_1) * ones(N,1) / N;
#        gv = (minibatch - v_state_1) * ones(N,1) / N;
#            
#        weight_cost = 1e-4; cw = weight_cost * obj.weight_v2h;
#        g = -[gw(:)-cw(:);gh(:);gv(:)];
#        return g
        pass
    
    # 计算重建误差(目标函数)
    def ffobject(self,x,i=None):
        pass
    
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