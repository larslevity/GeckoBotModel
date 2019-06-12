#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:15:05 2019

@author: ls
"""

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from Src.TrajectoryPlanner import search_tree as st
    from Src.Utils import plot_fun as pf
    from Src.Math import kinematic_model as model

    alpha = [90, 0, -90, 90, 0]
    eps = 90
    p1 = (0, 0)
    x, (mx, my), f = model.set_initial_pose(alpha, eps, p1)
    initial_pose = pf.GeckoBotPose(x, (mx, my), f)
    gait = pf.GeckoBotGait()
    gait.append_pose(initial_pose)

    ref = st.ReferenceGenerator('010')
    xref = (2, 3)

    def calc_dist(pose, xref):
        mx, my = pose.markers
        act_pos = np.r_[mx[1], my[1]]
        dpos = xref - act_pos
        return np.linalg.norm(dpos)


    act_pose = gait.poses[-1]
    x, y = act_pose.markers
    act_pos = (x[1], y[1])
    eps = act_pose.x[-1]

    alpha, feet, _,  pose_id = ref.get_next_reference(
            act_pos, eps, xref, act_pose, vis_dec=True)

    predicted_pose = model.predict_next_pose(
            [alpha, feet], act_pose.x, (x, y))

    predicted_pose = pf.GeckoBotPose(*predicted_pose)
    gait.append_pose(predicted_pose)


