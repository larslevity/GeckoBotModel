#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:06:41 2019

@author: ls
"""

import sys
from os import path
sys.path.insert(0, path.dirname(path.dirname(path.dirname(
        path.abspath(__file__)))))


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    from Src.Utils import plot_fun as pf
    from Src.Utils import save as my_save
    from Src.Math import kinematic_model as model

    from matplotlib import rc
    rc('text', usetex=True)
    # for Palatino and other serif fonts use:
#    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
#    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

    eps = 90
#    f_l, f_o, f_a = [.1, 1, 10]
    f_l, f_o, f_a = .2, 12.1, 6.1
    weight = [f_l, f_o, f_a]

    len_leg, len_tor = [9.1, 10.3]

    X1 = [60, 70, 80, 90]  # np.arange(70.01, 90.2, 10.)
#    X1 = [50, 80, 90]  # np.arange(70.01, 90.2, 10.)
    X2 = np.arange(-.5, .52, .2)

    n_cyc = 1
    take_every = 7
    sc = 10  # scale factor
    dx, dy = 3.5*sc, (3+2.5*(n_cyc-1 if n_cyc > 1 else 1))*sc/take_every

    def cut(x):
        return x if x > 0.001 else 0.001

    def alpha1(x1, x2, f, c1=1):
        alpha = [cut(45 - x1/2. - abs(x1)*x2/2. + (f[0])*x1*x2*c1),
                 cut(45 + x1/2. + abs(x1)*x2/2. + (f[1])*x1*x2*c1),
                 x1 + x2*abs(x1),
                 cut(45 - x1/2. - abs(x1)*x2/2. + (f[2])*x1*x2*c1),
                 cut(45 + x1/2. + abs(x1)*x2/2. + (f[3])*x1*x2*c1)
                 ]
        return alpha

    def alpha3(x1, x2, f, c1):
        alpha = [cut(45 - x1/2. - abs(x1)*x2/2. + x1*x2*c1),
                 cut(45 + x1/2. + abs(x1)*x2/2. + x1*x2*c1),
                 x1 + x2*abs(x1),
                 cut(45 - x1/2. - abs(x1)*x2/2. + x1*x2*c1),
                 cut(45 + x1/2. + abs(x1)*x2/2. + x1*x2*c1)
                 ]
        return alpha




# %%
    for x1_idx, x1 in enumerate(X1):
        print('x1:', x1)
        alpha = alpha3
        C1 = np.linspace(0, 2, 21)

        RESULT_DX = np.zeros((len(X2), len(C1)))
        RESULT_DY = np.zeros((len(X2), len(C1)))
        RESULT_DEPS = np.zeros((len(X2), len(C1)))
        RESULT_STRESS = np.zeros((len(X2), len(C1)))
        X_idx = np.zeros((len(X2), len(C1)))
        Y_idx = np.zeros((len(X2), len(C1)))
        GAITS = []
        for c1_idx, c1 in enumerate(C1):
            for x2_idx, x2 in enumerate(X2):
                f1 = [0, 1, 1, 0]
                f2 = [1, 0, 0, 1]
                if x2 < 0:
                    ref2 = [[alpha(-x1, x2, f2, c1), f2],
                            [alpha(x1, x2, f1, c1), f1]
                            ]
                else:
                    ref2 = [[alpha(x1, x2, f1, c1), f1],
                            [alpha(-x1, x2, f2, c1), f2]
                            ]
                ref2 += [ref2[0]]

                init_pose = pf.GeckoBotPose(
                        *model.set_initial_pose(
                                ref2[0][0], eps, (x2_idx*dx, c1_idx*dy),
                                len_leg=len_leg, len_tor=len_tor))
                gait = pf.predict_gait(
                        ref2, init_pose, weight, (len_leg, len_tor))

                (dxx, dyy), deps = gait.get_travel_distance()
                RESULT_DX[x2_idx][c1_idx] = dxx
                RESULT_DY[x2_idx][c1_idx] = dyy
                RESULT_DEPS[x2_idx][c1_idx] = deps
                print('(x1, x2, c1):', round(x1, 2), round(x2, 2),
                      round(c1, 2), ':', round(deps, 2))

#                Phi = gait.plot_phi()
#                plt.title('Phi')
#
                cumstress = gait.plot_stress()
                RESULT_STRESS[x2_idx][c1_idx] = cumstress
                plt.title('Inner Stress')
#
#                Alp = gait.plot_alpha()
#                plt.title('Alpha')
#
#                plt.figure('GeckoBotGait')

                X_idx[x2_idx][c1_idx] = x2_idx*dx
                Y_idx[x2_idx][c1_idx] = c1_idx*dy
                GAITS.append(gait)

# %% Save all the figs as png

        fig = plt.figure('GeckoBotGait')
        levels = [0, 25, 50, 100, 150, 200, 250, 350, 500, 1000, 1500, 2000, 3000]
        contour = plt.contourf(X_idx, Y_idx, RESULT_STRESS, alpha=.8,
                               cmap='OrRd', levels=levels)
        plt.colorbar(shrink=0.5, aspect=10,
                     label="cumulative stress over gait")
        plt.clim(0, 1500)

        surf = plt.contour(X_idx, Y_idx, RESULT_STRESS, levels=levels,
                           colors='black')
        plt.clabel(surf, levels, inline=True, fmt='%2.0f', fontsize=25)

        low_res = []
        gait_tex = ''
        for c1_idx in range(len(C1)):
            if c1_idx % take_every == 0 or c1_idx == len(C1)-1:
                low_res += [1]*len(X2)
            else:
                low_res += [0]*len(X2)
        for g_idx, gait in enumerate(GAITS):
            if low_res[g_idx] == 1:
                gait.plot_orientation(length=.5*sc)
#                gait.plot_gait()
                gait_tex = (gait_tex + '\n%%%%%%%\n'
                            + gait.get_tikz_repr(linewidth='.7mm', dashed=0))

        for xidx, x in enumerate(list(RESULT_DEPS)):
            for yidx, deps in enumerate(list(x)):
                if yidx % take_every == 0 or yidx == len(x)-1:
                    plt.text(X_idx[xidx][yidx], Y_idx[xidx][yidx]-2.2*sc,
                             '$'+str(round(deps, 1))+'^\\circ$',
                             ha="center", va="center",
                             fontsize=25,
                             bbox=dict(boxstyle="square",
                                       ec=(1., 0.5, 0.5),
                                       fc=(1., 0.8, 0.8),
                                       ))

        plt.xticks(X_idx.T[0], [round(x, 1) for x in X2])
        plt.yticks(Y_idx[0], [round(x, 2) for x in C1])
        plt.xlabel('steering $q_2$')
        plt.ylabel('additional bending for fixed legs $c_1$')
        plt.axis('scaled')
        plt.ylim((Y_idx[0][0]-30, Y_idx[0][-1]+20))
        plt.xlim((X_idx[0][0]-15, X_idx[-1][0]+15))
    #    plt.axis('auto')

        plt.grid()
        ax = fig.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        fname = '../../Out/analytic_model5ref/c1_analysis_q1_{}.tex'.format(x1)
        my_save.save_plt_as_tikz(fname,
                                 additional_tex_code=gait_tex,
                                 scale=.7,
                                 scope='scale=.1, opacity=.8')

#        fig.set_size_inches(10.5, 8)
#        fig.savefig(fname[:-3]+'png', transparent=True,
#                    dpi=300, bbox_inches='tight')
# %%
        fig.clear()  # clean figure

    plt.show()
