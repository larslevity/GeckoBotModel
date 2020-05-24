#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:04:21 2020

@author: ls

>
<

"""

import numpy as np
import matplotlib.pyplot as plt

from Src.TrajectoryPlanner import optimal_planner as op
from Src.Math import kinematic_model as model
from Src.Utils import plot_fun as pf

alp = [90, 0, -90, 90, 0]
eps = 90
ell = [1, 1, 1.2, 1, 1]
p1 = (0, 0)

x, marks, feet = model.set_initial_pose(
        alp, eps, p1, len_leg=ell[0], len_tor=ell[2])
feet = [1, 0, 0, 1]

initial_pose = pf.GeckoBotPose(x, marks, feet)
gait = pf.GeckoBotGait()
gait.append_pose(initial_pose)

Q = [(-80, 0.1), (-90, -0.5), (-30, -.4), (80, .4)]



t_move = 1.2
t_fix = .1
t_defix = .1



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
        feet = [0, 1, 1, 0] if alp[2] > 0 else [1, 0, 0, 1]  # ><
        if q1 < 0:  # switch fix for running backwards
            feet = [not(foot) for foot in feet]
        if feet != self.last_feet:  # switch fix
            pattern.append([self.last_alp, [1, 1, 1, 1], t[1]])  # fix
            pattern.append([self.last_alp, feet, t[2]])  # defix
        pattern.append([alp, feet, t[0]])  # move

        self.last_alp = alp
        self.last_feet = feet
        
        return pattern

osci = Oscillator(alp, feet)



for (q1, q2) in Q:
    pattern = osci.get_ref(q1, q2)
    for ref in pattern:
        print(ref)
        alp_ref, feet_ref, p_time = ref
    
        x, marks, f, constraint, cost = model.predict_next_pose(
            [alp_ref, feet_ref], x, marks, len_leg=1, len_tor=1.2)
        gait.append_pose(
            pf.GeckoBotPose(x, marks, f, constraint=constraint, cost=cost))

    gait.plot_gait()
    plt.show()
