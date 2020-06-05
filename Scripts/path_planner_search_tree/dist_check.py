#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 02:03:59 2020

@author: ls
"""


import numpy as np
import matplotlib.pyplot as plt

import sys
from os import path
sys.path.insert(0, path.dirname(path.dirname(path.dirname(
        path.abspath(__file__)))))

from Src.TrajectoryPlanner import state_machine as st
from Src.Utils import plot_fun as pf
from Src.Utils import save
from Src.Math import kinematic_model as model


tries = 1

MINDIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

sucess = {}

# %%

for mindist in MINDIST:
    sucess[mindist] = tries
    
    for try_idx in range(tries):

        alpha = [90, 0, -90, 90, 0]
        eps0 = 90
        p1 = (0, 0)
        ell = [9.1, 9.1, 10.3, 9.1, 9.1]
        f_l, f_o, f_a = 1, 1, 1
    
    
    
        x, (mx, my), f = model.set_initial_pose(alpha, eps0, p1, len_leg=ell[0], len_tor=ell[2])
        initial_pose = pf.GeckoBotPose(x, (mx, my), f,
                                       len_leg=ell[0], len_tor=ell[2])
        gait = pf.GeckoBotGait()
        gait.append_pose(initial_pose)
    
        ref = st.ReferenceGenerator('L')
        xref = (20, 30)
    
        def calc_dist(pose, xref):
            mx, my = pose.markers
            act_pos = np.r_[mx[1], my[1]]
            dpos = xref - act_pos
            return np.linalg.norm(dpos)
    
        def add_noise(alpha):
            return list(np.r_[alpha]+np.random.normal(0, 5, 5))
    
        i = 0
        
        XREF = [(0,0), (20, 30)]
        for xref in XREF:
            while calc_dist(gait.poses[-1], xref) > mindist:
                act_pose = gait.poses[-1]
                x, y = act_pose.markers
                act_pos = (x[1], y[1])
                eps = act_pose.x[-1]
        
                alpha, feet, _,  pose_id = ref.get_next_reference(
                        act_pos, eps, xref, act_pose, save_as_tikz=True, gait=gait,
                        show_dec=0)
                print('\n\npose ', i)
                print('pose:\t\t', pose_id, ' -- ', alpha)
                print('distance:\t', calc_dist(gait.poses[-1], xref))
                # NOISE
#                alpha = add_noise(alpha)
                
                if ':' not in pose_id or pose_id == 'crawling':
                    predicted_pose = model.predict_next_pose(
                            [alpha, feet], act_pose.x, (x, y),
                            len_leg=ell[0], len_tor=ell[2], f=[f_l, f_o, f_a])
            
                    predicted_pose = pf.GeckoBotPose(*predicted_pose)
                    gait.append_pose(predicted_pose)
                i += 1
                if i > 30:
                    print('Failed')
                    sucess[mindist] -= 1
                    break
    
    
    # %%
    
#        gait.plot_markers(1)
#    #    gait.plot_com()
#        for idx, xref in enumerate(XREF):
#            lab = 'goal '+str(idx) if idx > 0 else 'start'
#            st.draw_point_dir(xref, [0, 0], msize=15, label=lab, fontsize=30)
#        st.draw_point_arrow(p1, [np.cos(np.deg2rad(eps0))*10, np.sin(np.deg2rad(eps0))*10],
#                                 size=15, colp='orange')
#    
#        
#        plt.xlabel('$x$ position (cm)')
#        plt.ylabel('$y$ position (cm)')
#        
#        plt.axis('scaled')
#            
#        ax = plt.gca()
#        ax.spines['top'].set_visible(False)
#        ax.spines['left'].set_visible(False)
#        ax.spines['right'].set_visible(False)
#        ax.spines['bottom'].set_visible(False)
#        plt.grid()
    
    #    kwargs = {'extra_axis_parameters':
    #              {'x=.1cm', 'y=.1cm', 'anchor=origin', 'xmin=-55',
    #               'xmax=37','axis line style={draw opacity=0}',
    #               'ymin=-20, ymax=105', 'tick pos=left',}}
    #
    #    gait_str = gait.get_tikz_repr(dashed=0)
    #    save.save_plt_as_tikz('Out/pathplanner/gait.tex', gait_str,
    #                          scope='scale=.1', **kwargs)
    
# %%


def autolabel(rectdic):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for mode in rectdic:
        for key in rectdic[mode]:
            for rect in rectdic[mode][key]:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')


plt.figure('distcheck')
y = [sucess[key]*100/tries for key in MINDIST]
plt.bar(MINDIST, [100]*len(MINDIST), color='red', alpha=.5)
plt.bar(MINDIST, y, color='green', alpha=1)

plt.xlabel('minimum distance $\epsilon$ (cm)')
plt.ylabel('sucessful runs (%)')


# labels:
ax = plt.gca()
for idx, mindist in enumerate(MINDIST):
    percentage = y[idx]
    ax.annotate('{}'.format(round(percentage)),
                xy=(mindist, percentage),
                xytext=(0, -3 if percentage > 30 else 3),  # 3 points vertical offset
                textcoords="offset points",
                fontsize=16,
                ha='center', va='top' if percentage > 30 else 'bottom')


ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.yticks([])
MINDIST_ = MINDIST[:]
MINDIST_[-1] = 9.9999
plt.xticks(MINDIST_, [str(d) for d in MINDIST])
#plt.grid(axis='y')

kwargs = {'extra_axis_parameters':
          {'width=11cm', 'height=4cm'}}
save.save_plt_as_tikz('Out/dist_check.tex', **kwargs)


