#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 19:55:04 2019

@author: ls
"""

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from Src.Utils import plot_fun as pf
    from Src.Utils import save
    from Src.Math import kinematic_model as model

#%%

    alp = [0, 0, 0, 0, 0]
    eps = 90
    ell = [1, 1, 1, 1, 1]
    p1 = (1, 1)

    x, marks, f = model.set_initial_pose(
            alp, eps, p1, len_leg=ell[0], len_tor=ell[2])

    initial_pose = pf.GeckoBotPose(x, marks, f)
    gait = pf.GeckoBotGait()
    gait.append_pose(initial_pose)

    # REF
    refs = []
    step = 15
    for alp in np.arange(0,91,step):
        for bet in np.arange(-90,91,step):
            for gam in np.arange(0,91,step):
                refs.append([[0, alp, bet, gam, 0], [0, 1, 1, 0]])

    maxima = {'minx': 1, 'maxx': 1, 'miny': 1, 'maxy': 1}
    index = {'minx': 0, 'maxx': 0, 'miny': 0, 'maxy': 0}
    for idx, ref in enumerate(refs):

        x, marks, f, constraint, cost = model.predict_next_pose(
            ref, x, marks, len_leg=ell[0], len_tor=ell[2])
        gait.append_pose(
            pf.GeckoBotPose(x, marks, f, constraint=constraint, cost=cost))
        # check maxima:
        px, py = marks[0][1], marks[1][1]
        if px < maxima['minx']:
            maxima['minx'] = px
            index['minx'] = idx
        if px > maxima['maxx']:
            maxima['maxx'] = px
            index['maxx'] = idx
        if py < maxima['miny']:
            maxima['miny'] = py
            index['miny'] = idx
        if py > maxima['maxy']:
            maxima['maxy'] = py
            index['maxy'] = idx
        


#%%
#    gait.plot_gait()
    gait.plot_markers([1])
    plt.axis('off')
    gait_str = gait.poses[index['minx']].get_tikz_repr()
    gait_str = gait_str + gait.poses[index['maxx']].get_tikz_repr()
    gait_str = gait_str + gait.poses[index['miny']].get_tikz_repr()
    gait_str = gait_str + gait.poses[index['maxy']].get_tikz_repr()
    
    gait.poses[index['minx']].show_stats()
    gait.poses[index['maxx']].show_stats()
    gait.poses[index['maxy']].show_stats()
    
    save.save_plt_as_tikz('Out/workspace/test.tex', gait_str)


# %%
    mx, my = model._calc_coords2(x, marks, f)
    for x, y, i in zip(mx, my, [i for i in range(6)]):
        plt.plot(x, y, 'ro', markersize=10)
        plt.text(x, y+.1, str(i))
