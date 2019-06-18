# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:18:10 2019

@author: AmP
"""
import numpy as np


def rotate(vec, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.r_[c*vec[0]-s*vec[1], s*vec[0]+c*vec[1]]


if __name__ == "__main__":
    x = np.r_[[2, 2]]
    eps = np.deg2rad(45)
    x_ref = np.r_[[4, 4]]
    
    dx_R = rotate(x_ref - x, -eps)
    
    print(dx_R)
    
    