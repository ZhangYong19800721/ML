#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'实现s形函数，f(x)=1/(1+exp(-x))'

__author__ = '张勇,24452861@qq.com'

import numpy as np

# s形函数
def sigmoid(x):
    return 1/(1+np.exp(-x))
	
# 单元测试
if __name__ == '__main__':
    x=1
    y=sigmoid(x);
    print(y)