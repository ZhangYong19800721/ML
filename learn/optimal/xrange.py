#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""进退法确定搜索区间"""

__author__ = '张勇,24452861@qq.com'


def xrange(f, x0, h0, p):
    # 参数检查
    if p is None:  # 没有给出参数
        print('调用xrange函数时没有给出参数，将使用默认参数')
        p = {}

    if 'epsilon' not in p:  # 给出参数但是没有给出epsilon
        p['epsilon'] = 1e-6
        print('epsilon参数，将使用默认值%f' % p['epsilon'])

    f0 = f.ffobject(x0)  # 计算在x0位置的函数值
    x1 = x0 + h0  # 迭代到x1位置
    f1 = f.ffobject(x1)  # 计算在x1位置的函数值
    a = x0
    if f0 > f1:
        while f0 > f1:
            h0 = 2 * h0  # 扩大步长
            x1 = x0 + h0
            f1 = f.ffobject(x1)
        b = x1
    else:
        while f0 <= f1:
            h0 = 0.5 * h0  # 缩小步长
            if abs(h0) < p['epsilon']:
                break
            b = x1
            x1 = x0 + h0
            f1 = f.ffobject(x1)

    return a, b
