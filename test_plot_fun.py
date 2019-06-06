# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:45:38 2019

@author: AmP
"""

if __name__ == "__main__":
    from Src.Utils import plot_fun as pf
    from Src.Math import kinematic_model as model

    eps = 90
    F1 = (0, 0)
    

    ref = [[[0, 90, 90, 0, 90], [0, 1, 1, 0]],
           [[90, 90, 90, 90, 90], [1, 0, 0, 1]],
           ]

    ref2_ = [[
             [[45-gam/2.+x, 45+gam/2.+x, gam+x, 45-gam/2.+x, 45+gam/2.+x], [0, 1, 1, 0]],
             [[45+gam/2.+x, 45-gam/2.+x, -gam+x, 45+gam/2.+x, 45-gam/2.+x], [1, 0, 0, 1]]
            ] for gam, x in [(80,22), (80,22)]]
    ref2 = model.flat_list(ref2_)
#    print(ref2)
#    ref2 = ref

    init_pose = pf.GeckoBotPose(
            *model.set_initial_pose(ref2[0][0], eps, ref2[0][1]))
    gait = pf.predict_gait(ref2, init_pose)

    gait.plot_gait()
    gait.plot_markers()
    gait.plot_stress()
    gait.plot_travel_distance()
    gait.plot_orientation()
    gait.plot_alpha()
    gait.plot_phi()
    print(gait.get_travel_distance())



#    line_ani = gait.animate()
#    pf.save_animation(line_ani)
#    gait.save_as_tikz('test_plot_fun')
