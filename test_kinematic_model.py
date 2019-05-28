# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:36:34 2019

@author: AmP
"""

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from Src.Utils import plot_fun as pf
    from Src.Math import kinematic_model as model

    alp = [90, 0, -90, 90, 0]
    eps = 90
    ell = [1, 1, 1.2, 1, 1]
    p1 = (5, 10)

    x, marks, f = model.set_initial_pose(
            alp, eps, p1, len_leg=ell[0], len_tor=ell[2])

    initial_pose = pf.GeckoBotPose(x, marks, f)
    gait = pf.GeckoBotGait()
    gait.append_pose(initial_pose)

    # REF
    ref1 = [[20, 90, 90, 0, 120], [0, 1, 1, 0]]
    ref2 = [[100, 10, -70, 80, 2], [1, 0, 0, 1]]

    x, marks, f, stats = model.predict_next_pose(
        ref1, x, marks, len_leg=1, len_tor=1.2)
    gait.append_pose(
        pf.GeckoBotPose(x, marks, f, constraint=stats[0], cost=stats[1]))
    x, marks, f, stats = model.predict_next_pose(
        ref2, x, marks, len_leg=1, len_tor=1.2)
    gait.append_pose(
        pf.GeckoBotPose(x, marks, f, constraint=stats[0], cost=stats[1]))

    gait.plot_gait()
#    gait.plot_stress()
    gait.plot_markers()
#    gait.save_as_tikz('test')

    mx, my = model._calc_coords2(x, marks, f)
    for x, y, i in zip(mx, my, [i for i in range(6)]):
        plt.plot(x, y, 'ro', markersize=10)
        plt.text(x, y+.1, str(i))
