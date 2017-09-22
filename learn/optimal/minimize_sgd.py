#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'随机梯度下降'

__author__ = '张勇,24452861@qq.com'

import numpy as np

def minimize_sgd(f,x0,p=None):
    # 参数检查
    if p == None: # 没有给出参数
        print('调用minimize_sgd函数时没有给出参数集，将使用默认参数集')
        
    if not 'epsilon' in p: # 给出参数但是没有给出epsilon
        p['epsilon'] = 1e-3
        print('epsilon参数，将使用默认值%f' % p['epsilon'])
    
    if not 'max_it' in p: # 给出参数但是没有给出max_it
        p['max_it'] = 1e6
        print('max_it参数，将使用默认值%d' % p['max_it'])
    
    if not 'momentum' in p: # 给出参数但是没有给出momentum
        p['momentum'] = 0.9
        print('momentum参数，将使用默认值%f' % p['momentum'])
    
    if not 'learn_rate' in p: # 给出参数但是没有给出learn_rate
        p['learn_rate'] = 0.1
        print('learn_rate参数，将使用默认值%f' % p['learn_rate'])
    
    if not 'decay' in p: # 给出参数但是没有给出decay
        p['decay'] = 1 # 缺省情况下不降低学习速度
        print('decay参数，将使用默认值%f' % p['decay'])
    
    # 初始化
    x1 = x0
    inc_x = 0 # 参数的递增量
    
    # 开始迭代
    for it in range(p['max_it']):
        r  = p['learn_rate'] - (1 - 1/p['decay']) * p['learn_rate'] * it / p['max_it'] # 学习速度递减
        g1 = f.gradient(x1,it) # 计算梯度
        y1 = f.ffobject(x1,it) # 计算目标函数值
        ng1 = np.linalg.norm(g1) # 计算梯度模
        print('迭代次数:%d 学习速度:%f 目标函数:%f 梯度模:%f ' % (it,r,y1,ng1))
        if ng1 < p['epsilon']:
            break # 如果梯度足够小就结束迭代
        inc_x = p['momentum'] * inc_x - (1 - p['momentum']) * r * g1 # 向负梯度方向迭代，并使用动量参数
        x1 = x1 + inc_x # 更新参数值
    
    return x1,y1


# 单元测试
if __name__ == '__main__':
    pass