# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:59:07 2019

@author: AmP
"""
import numpy as np


def rotate(vec, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.r_[c*vec[0]-s*vec[1], s*vec[0]+c*vec[1]]


def normalize(vec):
    x, y = vec
    leng = np.sqrt(x**2 + y**2)
    return np.r_[x/leng, y/leng]


def calc_angle(vec1, vec2, rotate_angle=0., jump=0):
    theta = np.radians(rotate_angle)
    vec1 = rotate(vec1, theta)
    x1, y1 = vec1  # normalize(vec1)
    x2, y2 = vec2  # normalize(vec2)
    phi1 = np.arctan2(y1, x1)
    vec2 = rotate([x2, y2], -phi1+jump)
    phi2 = np.degrees(np.arctan2(vec2[1], vec2[0]) - jump)
    alpha = -phi2
    return alpha
