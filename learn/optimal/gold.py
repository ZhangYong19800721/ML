#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""黄金分割法进行一维精确线搜索"""

__author__ = '张勇,24452861@qq.com'


def gold(f, a, b, p):
    # 参数检查
    if p is None:  # 没有给出参数
        print('调用gold函数时没有给出参数，将使用默认参数')
        p = {}

    if 'epsilon' not in p:  # 给出参数但是没有给出epsilon
        p['epsilon'] = 1e-6
        print('epsilon参数，将使用默认值%f' % p['epsilon'])

    # 使用黄金分割法进行一维精确线搜索
    g = (5 ** 0.5 - 1) / 2
    ax, bx = a + (1 - g) * (b - a), a + g * (b - a)
    f_ax, f_bx = f.ffobject(ax), f.ffobject(bx)

    while b - a > p['epsilon']:
        if f_ax > f_bx:
            a = ax
            ax, f_ax = bx, f_bx
            bx = a + g * (b - a)
            f_bx = f.ffobject(bx)
        else:
            b = bx
            bx, f_bx = ax, f_ax
            ax = a + (1 - g) * (b - a)
            f_ax = f.ffobject(ax)

    if f_ax > f_bx:
        nx, ny = bx, f_bx
    else:
        nx, ny = ax, f_ax

    return nx, ny
