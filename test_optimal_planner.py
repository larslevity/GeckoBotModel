# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:42:20 2019

@author: AmP
"""

from Src.TrajectoryPlanner import optimal_planner as op
from Src.Math import kinematic_model as model
from Src.Utils import plot_fun as pf

alp = [90, 0, -90, 90, 0]
eps = 0
ell = [1, 1, 1.2, 1, 1]
p1 = (0, 0)

x, marks, feet = model.set_initial_pose(
        alp, eps, p1, len_leg=ell[0], len_tor=ell[2])
feet = [1, 0, 0, 1]

initial_pose = pf.GeckoBotPose(x, marks, feet)
gait = pf.GeckoBotGait()
gait.append_pose(initial_pose)

x_ref = (10, 10)

xbar = op.xbar(x_ref, p1, eps)

alp_ref, feet_ref = op.optimal_planner(xbar, alp, feet,
                                       n=2, dist_min=.1, show_stats=0)



x, marks, f, constraint, cost = model.predict_next_pose(
    [alp_ref, feet_ref], x, marks, len_leg=1, len_tor=1.2)
gait.append_pose(
    pf.GeckoBotPose(x, marks, f, constraint=constraint, cost=cost))

gait.plot_gait()
