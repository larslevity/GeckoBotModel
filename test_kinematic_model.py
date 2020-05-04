# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:36:34 2019

@author: AmP
"""

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    from Src.Utils import plot_fun as pf
    from Src.Math import kinematic_model as model

    alp = [5, 110, 120, 10, 110]
    eps = 20
    ell = [3, 3, 3.5, 3, 3]
    p1 = (5, 10)

    x, marks, f = model.set_initial_pose(
            alp, eps, p1, len_leg=ell[0], len_tor=ell[2])

    initial_pose = pf.GeckoBotPose(x, marks, f)
    gait = pf.GeckoBotGait()
    gait.append_pose(initial_pose)

    # REF
    ref1 = [[70, 0, -45, 70, 0], [1, 0, 0, 1]]
    ref2 = [alp, [0, 1, 1, 0]]

    tic = time.time()
    x, marks, f, constraint, cost = model.predict_next_pose(
        ref1, x, marks, len_leg=ell[0], len_tor=ell[2])
    print('ellapsed time:', time.time()-tic, 's')
    gait.append_pose(
        pf.GeckoBotPose(x, marks, f, constraint=constraint, cost=cost))

    tic = time.time()
    x, marks, f, constraint, cost = model.predict_next_pose(
        ref2, x, marks, len_leg=ell[0], len_tor=ell[2])
    print('ellapsed time:', time.time()-tic, 's')
    gait.append_pose(
        pf.GeckoBotPose(x, marks, f, constraint=constraint, cost=cost))

    gait.plot_gait()
#    gait.plot_stress()
    gait.plot_markers()
#    gait.save_as_tikz('test')

    mx, my = model._calc_coords2(x, marks, f)
    for x, y, i in zip(mx, my, [i for i in range(6)]):
        plt.plot(x, y, 'ro', markersize=10)
        plt.text(x, y+.1, str(i))
