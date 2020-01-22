# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:50:26 2020

@author: AmP
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import rc
rc('text', usetex=True)

from os import path
sys.path.insert(0, path.dirname(path.dirname(path.dirname(
        path.abspath(__file__)))))

from Src.Utils import plot_fun as pf
from Src.Math import kinematic_model as model

DEPS_EXP = np.array([[ 21.08      ,  24.92631579,  27.01428571,  25.88888889,
         21.80740741],
       [ 14.22608696,  19.45      ,  18.78823529,  15.125     ,
          6.15384615],
       [  5.13      ,   5.88333333,   4.5       ,   4.98      ,
          0.4       ],
       [ -1.07      ,  -3.9875    ,  -7.01428571, -10.35294118,
        -10.4       ],
       [ -9.575     , -15.05454545, -18.47272727, -17.21666667,
        -15.4875    ],
       [-18.55454545, -28.55789474, -30.72222222, -26.34166667,
        -24.75555556]])

eps = 90
len_leg, len_tor = [9.1, 10.3]

for f_l in np.arange(.1, 2, 0.1):
    for f_o in np.arange(5, 20, 1):
        for f_a in np.arange(1, 10, 1):

            weight = [f_l, f_o, f_a]
            modelstr = 'f_l, f_0, f_a = ' + str(f_l) + ', ' + str(f_o) + ', ' + str(f_a)
            print(modelstr)

            X1 = np.arange(50.01, 90.2, 10.)   # GaitLawExp
            X2 = np.arange(-.5, .52, .2)  # GaitLawExp

            n_cyc = 1     # doc: 1  # poster IROS: 2
            and_half = True  # doc:True, #IROS:FALSE

            sc = 10  # scale factor
            dx, dy = 2.8*sc, (4.5)*sc

            def cut(x):
                return x if x > 0.001 else 0.001

            def alpha2(x1, x2, f, c1=1):
                alpha = [cut(45 - x1/2. - abs(x1)*x2/2. + x1*x2*c1),
                         cut(45 + x1/2. + abs(x1)*x2/2. + x1*x2*c1),
                         x1 + x2*abs(x1),
                         cut(45 - x1/2. - abs(x1)*x2/2. + x1*x2*c1),
                         cut(45 + x1/2. + abs(x1)*x2/2. + x1*x2*c1)
                         ]
                return alpha

            alpha = alpha2

            RESULT_DX = np.zeros((len(X2), len(X1)))
            RESULT_DY = np.zeros((len(X2), len(X1)))
            RESULT_DEPS = np.zeros((len(X2), len(X1)))
            RESULT_STRESS = np.zeros((len(X2), len(X1)))
            X_idx = np.zeros((len(X2), len(X1)))
            Y_idx = np.zeros((len(X2), len(X1)))
            GAITS = []

            for x1_idx, x1 in enumerate(X1):
                for x2_idx, x2 in enumerate(X2):
                    X_idx[x2_idx][x1_idx] = x2_idx*dx
                    Y_idx[x2_idx][x1_idx] = x1_idx*dy

                    f1 = [0, 1, 1, 0]
                    f2 = [1, 0, 0, 1]
                    if x2 < 0:
                        ref2 = [[alpha(-x1, x2, f2), f2],
                                [alpha(x1, x2, f1), f1]
                                ]
                    else:
                        ref2 = [[alpha(x1, x2, f1), f1],
                                [alpha(-x1, x2, f2), f2]
                                ]
                    ref2 = ref2*n_cyc
                    if and_half:
                        ref2 += [ref2[0]]

                    init_pose = pf.GeckoBotPose(
                            *model.set_initial_pose(ref2[0][0], eps,
                                                    (x2_idx*dx, x1_idx*dy),
                                                    len_leg=len_leg,
                                                    len_tor=len_tor))
                    gait = pf.predict_gait(ref2, init_pose, weight,
                                           (len_leg, len_tor))

                    (dxx, dyy), deps = gait.get_travel_distance()
                    RESULT_DX[x2_idx][x1_idx] = dxx
                    RESULT_DY[x2_idx][x1_idx] = dyy
                    RESULT_DEPS[x2_idx][x1_idx] = deps
                    print('(x2, x1):', round(x2, 2), round(x1, 1), ':',
                          round(deps, 2))

                    GAITS.append(gait)

            Sim_err = np.linalg.norm(RESULT_DEPS-DEPS_EXP*n_cyc)
            print('Sim Err: ', Sim_err)

        # %% Save all GAIT/DEPS as png

            fig = plt.figure('GeckoBotGait')
            levels = np.arange(-65, 66, 5)
            contour = plt.contourf(X_idx, Y_idx, RESULT_DEPS, alpha=1,
                                   cmap='RdBu_r', levels=levels)
            for gait in GAITS:
                gait.plot_gait()
                gait.plot_orientation(length=.5*sc)

            for xidx, x in enumerate(list(RESULT_DEPS)):
                for yidx, deps in enumerate(list(x)):
                    plt.text(X_idx[xidx][yidx], Y_idx[xidx][yidx]-2.2*sc,
                             round(deps, 2),
                             ha="center", va="bottom",
                             bbox=dict(boxstyle="square",
                                       ec=(1., 0.5, 0.5),
                                       fc=(1., 0.8, 0.8),
                                       ))

        #        plt.colorbar(shrink=0.5, aspect=10,
        #                     label="cumulative stress over gait")
        #        plt.clim(0, 200)

            plt.xticks(X_idx.T[0], [round(x, 2) for x in X2])
            plt.yticks(Y_idx[0], [int(x) for x in X1])
            plt.xlabel('steering $q_2$')
            plt.ylabel('step length $q_1$')
            plt.axis('scaled')

            plt.grid()
            ax = fig.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            fig.set_size_inches(10.5, 8)
            fig.savefig('../../Out/analytic_model10/simerr='+str(round(Sim_err, 2))+'gait_deps_{}.png'.format(modelstr),
                        transparent=True,
                        dpi=300, bbox_inches='tight',)
    # %%
            fig.clear()  # clean figure
# %%

    plt.show()
