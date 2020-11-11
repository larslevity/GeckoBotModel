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
        [[1, 90, 90, 1, 90], [0, 1, 1, 0], 'pose1'],
#        [[1, 90, 90, 1, 90], [1, 1, 1, 1]],
        [[1, 90, 90, 1, 90], [1, 0, 0, 1]],
        [[90, 1, -90, 90, 1], [1, 0, 0, 1], 'pose2'], 
#        [[90, 1, -90, 90, 1], [1, 1, 1, 1]],
        [[90, 1, -90, 90, 1], [0, 1, 1, 0]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0], 'pose1'],
            ]

    elemtary_patterns['crawling_left_curve'] = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0], 'pose1'],
        [[45, 45, 1, 45, 45], [1, 0, 0, 1]],
        [[45, 45, 90, 45, 45], [1, 1, 0, 0]],
        [[45, 45, 1, 45, 45], [0, 0, 1, 1]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0], 'pose1'],
            ]

#    elemtary_patterns['curve_left'] = [
#        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
#        [[90, 1, -90, 90, 1], [1, 0, 0, 1]],
#        [[1, 40, 10, 10, 60], [0, 1, 1, 0]],
#        [[90, 1, -90, 90, 1], [1, 0, 0, 1]],
#        [[1, 90, 90, 1, 90], [0, 1, 1, 0]],
#            ]

    elemtary_patterns['curve_right'] = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0], 'pose1'],
        [[1, 90, 90, 1, 90], [1, 0, 0, 1]],
        [[40, 1, -10, 60, 10], [1, 0, 0, 1]],
        [[40, 1, -10, 60, 10], [0, 1, 1, 1]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0], 'pose1'],
            ]

    elemtary_patterns['curve_right_tight'] = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0], 'pose1'],
        [[1, 90, 90, 1, 90], [1, 0, 0, 1]],
        [[48, 104, 114, 27, 124], [0, 1, 1, 0]],
        [[48, 104, 114, 27, 124], [1, 0, 0, 1]],
        [[1, 72, 70, 1, 55], [1, 0, 0, 1]],
        [[1, 72, 70, 1, 55], [0, 1, 1, 0]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0], 'pose1'],
            ]
#
#
    elemtary_patterns['curve_right_super_tight'] = [
        [[1, 90, 90, 1, 90], [0, 1, 1, 0], 'pose1'],
        [[1, 90, 90, 1, 90], [1, 0, 0, 1]],
        [[50, 30, 90, 30, 150], [1, 0, 0, 1]],
        [[50, 30, 90, 30, 150], [0, 1, 1, 0]],
        [[124, 164, 152, 62, 221], [0, 1, 1, 0]],
        [[124, 164, 152, 62, 221], [1, 0, 0, 1]],
        [[30, 90, 80, 10, 10], [1, 0, 0, 1]],
        [[30, 90, 80, 10, 10], [0, 1, 1, 0]],
        [[1, 90, 90, 1, 90], [0, 1, 1, 0], 'pose1'],
            ]
    
    
    for key in elemtary_patterns:
        gait = pf.GeckoBotGait()
        x, marks, f = model.set_initial_pose(
            alp_0, eps_0, (0, 0), len_leg=1, len_tor=1.2)
        for ref in elemtary_patterns[key]:
            try:
                posename = ref[2]
                ref = ref[:2]
            except IndexError:
                posename = None
            x, marks, f, constraint, cost = model.predict_next_pose(
                    ref, x, marks, len_leg=1, len_tor=1.2, f=f_weights)
            gait.append_pose(
                    pf.GeckoBotPose(x, marks, f, constraint=constraint, name=posename))
            
        
 
    # %%
        plt.figure('python')
        gait.plot_gait()
        gait.plot_markers([1])
        plt.axis('off')
        plt.show()
        
    
    # %% SAVE AS TIKZ
        option = """
pose1/.style = {shape=circle, draw, align=center, top color=white, bottom color=blue!40, minimum width=1.9cm, opacity=.5},
pose2/.style = {shape=circle, draw, align=center, top color=white, bottom color=red!40, minimum width=1.9cm, opacity=.5},
pose3/.style = {shape=circle, draw, align=center, top color=white, bottom color=yellow!40},"""

        extra_axis_parameters = {'anchor=origin', 'disabledatascaling', 'x=1cm', 'y=1cm',
                   'axis line style={draw opacity=0}',
                   'clip mode=individual'}

        plt.figure('tikz')
        xshift = 4
#        gait.plot_travel_distance(shift=[xshift*(len(gait.poses)-1)+4,-.5], w=.05, size=5, colp='k')
        gait.plot_orientation(poses=[0],w=.05, size=5)
        gait.plot_orientation(poses=[-1], shift=[xshift*(len(gait.poses)-1),0], w=.05, size=5)
        plt.axis('off')
        for idx in range(len(gait.poses)):
            plt.annotate(str(idx), [xshift*idx+.5, -2.5], size=35)
        
        gait_str = gait.get_tikz_repr(shift=xshift, R=.15, dashed=0, reverse_col=-1,
                                      linewidth='.4mm')
        save.save_plt_as_tikz('Out/elemtary_patterns/{}.tex'.format(key),
                              gait_str, additional_options=option, scale=.7,
                              extra_axis_parameters=extra_axis_parameters)

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
