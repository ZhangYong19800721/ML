#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""随机梯度下降(ADAM)"""

__author__ = '张勇,24452861@qq.com'

import numpy as np


def minimize_adam(f, x0, p=None):
    # 参数检查
    if p is None:  # 没有给出参数
        print('调用minimize_adam函数时没有给出参数集，将使用默认参数集')
        p = {}

    if 'epsilon' not in p:  # 给出参数但是没有给出epsilon
        p['epsilon'] = 1e-8
        print('epsilon参数，将使用默认值%f' % p['epsilon'])

    if 'max_it' not in p:  # 给出参数但是没有给出max_it
        p['max_it'] = 1e6
        print('max_it参数，将使用默认值%d' % p['max_it'])

    if 'beta1' not in p:  # 给出参数但是没有给出beta1
        p['beta1'] = 0.9
        print('beta1参数，将使用默认值%f' % p['beta1'])

    if 'beta2' not in p:  # 给出参数但是没有给出beta2
        p['beta2'] = 0.999
        print('beta2参数，将使用默认值%f' % p['beta2'])

    if 'learn_rate' not in p:  # 给出参数但是没有给出learn_rate
        p['learn_rate'] = 0.001
        print('learn_rate参数，将使用默认值%f' % p['learn_rate'])

    # 初始化
    m = 0  # 初始化第一个递增向量
    v = 0  # 初始化第二个递增向量
    x1 = x0  # 起始点
    f1 = f.ffobject(x1)  # 计算目标函数值

    # 开始迭代
    for it in range(1, p['max_it']):
        g1 = f.gradient(x1, it)  # 计算梯度
        f1 = f.ffobject(x1, it)  # 计算目标函数值
        ng1 = np.linalg.norm(g1)  # 计算梯度模
        print('迭代次数:%d 学习速度:%f 目标函数:%f 梯度模:%f ' % (it, p['learn_rate'], f1, ng1))
        if ng1 < p['epsilon']:
            break  # 如果梯度足够小就结束迭代

        m = p['beta1'] * m + (1 - p['beta1']) * g1  # 更新第1个增量向量
        v = p['beta2'] * v + (1 - p['beta2']) * g1 ** 2  # 更新第2个增量向量
        mb = m / (1 - p['beta1'] ** it)  # 对第1个增量向量进行修正
        vb = v / (1 - p['beda2'] ** it)  # 对第2个增量向量进行修正
        x1 -= p['learn_rate'] * mb / (vb ** 0.5 + p['epsilon'])

    # 返回
    return x1, f1


# 单元测试
if __name__ == '__main__':
    pass
