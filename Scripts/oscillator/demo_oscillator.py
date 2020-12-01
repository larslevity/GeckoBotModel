#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:16:28 2020

@author: ls
"""


import numpy as np
import matplotlib.pyplot as plt


import sys
from os import path
sys.path.insert(0, path.dirname(path.dirname(path.dirname(
        path.abspath(__file__)))))

from Src.TrajectoryPlanner import optimal_planner as op
from Src.Math import kinematic_model as model
from Src.Utils import plot_fun as pf
from Src.Utils import save


# %%
#
#f_l = .1
#f_o = 1
#f_a = 10

f_l = 89.      # factor on length objective
f_o = 10     # .0003     # factor on orientation objective
f_a = 5.9        # factor on angle objective
weights = [f_l, f_o, f_a]


## init pose
#alp0 = op.alpha(-80, -.5)
alp0 = [90, 0, -90, 90, 0]
eps0 = 180
ell = [9.1, 9.1, 10.3, 9.1, 9.1]
p1 = (0, 0)

# Q ref
Q = [(-80, .5), (-80, .5), (-80, .5), (-80, .5), (80, -.5), (80, -.5), (80, -.5), (80, -.5), (80, -.5)]


# times
t_move = 1.2
t_fix = .1
t_defix = .1


# %%

class Oscillator(object):
    def __init__(self, alp=[10, 10, 10, 10, 10], feet=[1, 0, 0, 1]):
        self.last_alp = alp
        self.last_feet = feet

    def get_ref(self, q1, q2, t=[1.2, .1, .1]):
        """
        1. generate reference according to input
        2. Check fixation
            a. fix
            b. defix
        3. move
        """
        pattern = []
        
        alp = op.alpha(abs(q1)*np.sign(self.last_alp[2]*-1), q2)
        feet = [0, 1, 1, 0] if alp[2]*q1 > 0 else [1, 0, 0, 1]  # ><

#        if feet != self.last_feet:  # switch fix
#            pattern.append([self.last_alp, [1, 1, 1, 1], t[1]])  # fix
#            pattern.append([self.last_alp, feet, t[2]])  # defix
        pattern.append([alp, feet, t[0]])  # move

        self.last_alp = alp
        self.last_feet = feet
        
        return pattern



# %% with Sim Model

x, marks, feet = model.set_initial_pose(
        alp0, eps0, p1, len_leg=ell[0], len_tor=ell[2])
feet = [1, 0, 0, 1]

initial_pose = pf.GeckoBotPose(x, marks, feet)
gait = pf.GeckoBotGait()
gait.append_pose(initial_pose)

osci = Oscillator(alp0, feet)

for (q1, q2) in Q:
    pattern = osci.get_ref(q1, q2)
    for ref in pattern:
        print(ref)
        alp_ref, feet_ref, p_time = ref
    
        x, marks, f, constraint, cost = model.predict_next_pose(
            [alp_ref, feet_ref], x, marks, len_leg=ell[0], len_tor=ell[2], f=weights)
        gait.append_pose(
            pf.GeckoBotPose(x, marks, f, constraint=constraint, cost=cost))

#gait.plot_gait()


gait.plot_markers(1)
gait.plot_orientation(length=7, w=.4, size=3, colp='orange')


# %%

#plt.xlabel('$x$ position (cm)')
#plt.ylabel('$y$ position (cm)')

plt.axis('scaled')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.grid()

kwargs = {'extra_axis_parameters':
          {'x=.15cm', 'y=.15cm', 'anchor=origin',
           'xmin=-30', 'xmax=20',
           'axis line style={draw opacity=0}', # 'hide axis',
           'ymin=-20, ymax=30',
           'tick pos=left',}}

gait_str = gait.get_tikz_repr(dashed=0, linewidth='1mm', R=.8)
save.save_plt_as_tikz('Out/oscillator_demo.tex', gait_str,
                      scope='scale=.15', **kwargs)



plt.show()

# %% Q plot

q1 = [qi[0] for qi in Q]
q2 = [qi[1] for qi in Q]

plt.figure('Q1Q2')
ax1 = plt.gca()
ax2 = ax1.twinx()

ax1.plot(q1, 'blue')
ax2.plot(q2, 'red')

ax1.grid()
ax2.set_ylim(-.56, .56)
ax2.set_yticks([-.5, 0, .5], ['-.5', '0', '.5'])
ax1.set_ylim(-90, 90)
ax1.set_ylabel('step length $q_1$ ($^\\circ^$)', color='blue')
ax2.set_ylabel('steering q2 (1)', color='red')
ax1.set_xlabel('steps (1)')
ax2.tick_params('y', colors='red')
ax1.tick_params('y', colors='blue')

save.save_plt_as_tikz('Out/oscillator_demo_q.tex')


