#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'抽样函数，当随机数小于抽样概率时取1，当随机数大于抽样概率时取0'

__author__ = '张勇,24452861@qq.com'

import numpy as np

def sample(x):
    return np.rand(x.shape) < x;

