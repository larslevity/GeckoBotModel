#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:07:33 2020

@author: ls
"""

import matplotlib.pyplot as plt
from scipy.optimize import minimize

import sys
from os import path
sys.path.insert(0, path.dirname(path.dirname(path.dirname(
        path.abspath(__file__)))))

from Src.Utils import plot_fun as pf
from Src.Math import kinematic_model as model
from Src.Utils import save



if __name__ == "__main__":
    
    alp_0 = [1, 90, 90, 1, 90]
    eps_0 = 90
        
    
    f_len = 10
    f_ang = 100
    f_ori = 10
    f_weights = [f_len, f_ori, f_ang]
    
    elemtary_patterns = {}
    
    elemtary_patterns['straight'] = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
        [[90, 1, -90, 90, 1], [1, 0, 0, 1]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
            ]

    elemtary_patterns['curve_spezial'] = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
        [[45, 45, 1, 45, 45], [1, 0, 0, 1]],
        [[45, 45, 90, 45, 45], [1, 1, 0, 0]],
        [[45, 45, 1, 45, 45], [0, 0, 1, 1]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
            ]

    elemtary_patterns['curve_left'] = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
        [[90, 1, -90, 90, 1], [1, 0, 0, 1]],
        [[1, 40, 10, 10, 60], [0, 1, 1, 0]],
        [[90, 1, -90, 90, 1], [1, 0, 0, 1]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
            ]

    elemtary_patterns['curve_right'] = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
        [[40, 1, -10, 60, 10], [1, 0, 0, 1]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
            ]

    elemtary_patterns['curve_right_tight'] = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
        [[48, 104, 114, 27, 124], [0, 1, 1, 0]],
        [[1, 72, 70, 1, 55], [1, 0, 0, 1]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
            ]


    elemtary_patterns['curve_right_super_tight'] = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
        [[50, 30, 90, 30, 150], [1, 0, 0, 1]],
        [[124, 164, 152, 62, 221], [0, 1, 1, 0]],
        [[30, 90, 80, 10, 10], [1, 0, 0, 1]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
            ]
    
    
    for key in elemtary_patterns:
        gait = pf.GeckoBotGait()
        x, marks, f = model.set_initial_pose(
            alp_0, eps_0, (0, 0), len_leg=1, len_tor=1.2)
        for ref in elemtary_patterns[key]:
            x, marks, f, constraint, cost = model.predict_next_pose(
                    ref, x, marks, len_leg=1, len_tor=1.2, f=f_weights)
            gait.append_pose(
                    pf.GeckoBotPose(x, marks, f, constraint=constraint, cost=cost))
            
        
 
    # %%
        gait.plot_gait()
        gait.plot_markers([1])
        plt.axis('off')
        plt.show()
    
    # %% SAVE AS TIKZ
        gait.plot_markers(1)
        plt.axis('off')
        gait_str = gait.get_tikz_repr(shift=3)
        save.save_plt_as_tikz('Out/elemtary_patterns/{}.tex'.format(key),
                              gait_str)

    # %%
    
    
#    calc_gait_and_draw_tikz(curve_right_super_tight, 'curve_right_super_tight')
#
#    example = [
#            curve_right_super_tight,
#            curve_right,
#            curve_right,
#            straight,
#            straight,
#            straight,
#            curve_left,
#            curve_left,
#            curve_left,
#            curve_left
#             ]
##    draw_multiple_gaits_in_a_row(example, 'example', eps0=110)
#    example2 = [
#            straight,
#            straight,
#            curve_right,
#            straight,
#            curve_right,
#            straight,
#            straight,
#            curve_right_tight,
#            curve_right_super_tight,
#            straight,
#            curve_left,
#            straight,
#            straight,
#            straight,
#            straight,
#            straight,
#            curve_right,
#            curve_right_tight,
#            straight,
#            curve_right_tight,
#            straight
#             ]
#
#    draw_multiple_gaits_in_a_row(example2, 'example2', eps0=110)
