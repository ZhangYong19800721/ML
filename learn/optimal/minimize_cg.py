#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""共轭梯度下降"""

__author__ = '张勇,24452861@qq.com'

import numpy as np
from learn.optimal.singlex import singlex

def minimize_cg(f,x0,p=None):
    if p is None: # 没有给出参数
        print('调用minimize_cg函数时没有给出参数集，将使用默认参数集')
        p = {}
    
    if not 'epsilon' in p: # 给出参数但是没有给出epsilon
        p['epsilon'] = 1e-6 
        print('epsilon参数，将使用默认值%f' % p['epsilon'])
    
    if not 'max_it' in p: # 给出参数但是没有给出max_it
        p['max_it'] = 1e6
        print('max_it参数，将使用默认值%d',p['max_it'])
    
    if not 'reset' in p: # 给出参数但是没有给出reset
        p['reset'] = 500
        print('reset参数，将使用默认值%f',p['reset'])
    
    if not 'gold' in p: # 给出参数但是没有给出gold
        p['gold'] = None
        print('gold参数，将使用默认值%f',p['gold'])

    # 计算起始位置的函数值、梯度、梯度模
    x1 = x0
    y1 = f.ffobject(x1) # 计算函数值
    g1 = f.gradient(x1) # 计算梯度
    ng1 = np.linalg.norm(g1) # 计算梯度模
    if ng1 < p['epsilon']: # 如果梯度模足够小，直接返回
        return x1,y1

    # 迭代寻优
    d1 = -g1 # 初始搜索方向为负梯度方向
    for it in range(int(p['max_it'])):
        if ng1 < p['epsilon']: # 如果梯度模足够小，返回
            return x1,y1

        # 沿d1方向线搜索
		# 黄金分割法进行一维精确线搜索
        fs = singlex(f, x1, d1) # 包装为单变量函数
        a,b = xrange(fs, 0, 1, p['gold']) # 确定搜索区间
        y2, lamda = gold(fs, a, b, p['gold'])
        x2 = x1 + lamda * d1
    
        if (it % p['reset'] == 0 and it > 0) or (y1 <= y2): # 到达重置点或d1方向不是一个下降方向
            d1 = -g1 # 设定搜索方向为负梯度方向
            # 黄金分割法进行一维精确线搜索
			fs = singlex(f, x1, d1) # 包装为单变量函数
            a, b = xrange(fs, 0, 1, p['gold']) # 确定搜索区间
            y2, lamda = gold(fs, a, b, p['gold'])
            x2 = x1 + lamda * d1
   
            g2 = f.gradient(x2)
            d2 = -g2
            ng2 = norm(g2) # 迭代到新的位置x2，并计算函数值、梯度、搜索方向、梯度模
            x1,d1,g1,y1,ng1 = x2,d2,g2,y2,ng2
            print('目标函数:%f 迭代次数:%d 梯度模:%f ' % (y1, it, ng1))
            continue

        g2 = f.gradient(x2)
        ng2 = norm(g2) # 计算x2处的梯度和梯度模
        beda = g2.T.dot(g2-g1) / g1.T.dot(g1)
        d2 = -g2 + beda * d1 # 计算x2处的搜索方向d2
        x1,d1,g1,y1,ng1 = x2,d2,g2,y2,ng2
        print('目标函数:%f 迭代次数:%d 梯度模:%f ' % (y1, it, ng1))
		
    return x1,y1