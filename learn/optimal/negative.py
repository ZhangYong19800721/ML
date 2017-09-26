#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""包裹器，将f函数包裹为一个负函数"""

__author__ = '张勇,24452861@qq.com'


class negative(object):
    def __init__(self, f):
        self.f = f

    def ffobject(self, x):
        y = -self.f.ffobject(x)
        return y

    def gradient(self,x):
        y = -self.f.gradient(x)
        return y
