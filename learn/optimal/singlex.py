#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""包裹器，将f函数包裹为一个单变量函数"""

__author__ = '张勇,24452861@qq.com'


class singlex(object):
    def __init__(self, f, x0, d0):
        self.f = f
        self.x0 = x0
        self.d0 = d0

    def ffobject(self, x):
        y = self.f.ffobject(self.x0 + x * self.d0)
        return y
