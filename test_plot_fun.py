# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:45:38 2019

@author: AmP
"""

if __name__ == "__main__":
    from Src.Utils import plot_fun as pf
    from Src.Math import kinematic_model as model

    alpha = [0, 0, -0, 0, 0]
    eps = 90
    F1 = (0, 0)
    init_pose = pf.GeckoBotPose(*model.set_initial_pose(alpha, eps, F1))

    ref = [[[0, 90, 90, 0, 90], [0, 1, 1, 0]],
           [[10, 90, 90, 0, 90], [0, 1, 1, 0]],
           [[30, 90, -90, 0, 90], [0, 1, 1, 0]]]

    gait = pf.predict_gait(ref, init_pose)
    gait.save_as_tikz('test_plot_fun')
    gait.plot_gait()
    gait.plot_travel_distance()
    print(gait.get_travel_distance())

    line_ani = gait.animate()
    pf.save_animation(line_ani)
