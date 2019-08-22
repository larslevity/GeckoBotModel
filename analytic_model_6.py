#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 01:25:51 2019

@author: ls
"""

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm

    from Src.Utils import plot_fun as pf
    from Src.Math import kinematic_model as model

    from matplotlib import rc
    rc('text', usetex=True)
    # for Palatino and other serif fonts use:
#    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
#    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

    eps = 90
    f_l, f_o, f_a = 10, 1, 10
    weight = [f_l, f_o, f_a]

    X1 = np.arange(0.001, 90.2, 10.)
    X2 = np.arange(-.5, .52, .1)

    n_cyc = 2
    and_half = False

    dx, dy = 3.5, 3+2.5*(n_cyc-1 if n_cyc > 1 else 1)

    def cut(x):
        return x if x > 0.001 else 0.001

    def alpha1(x1, x2, f):
        alpha = [cut(45 - x1/2. - abs(x1)*x2/2. + (f[0])*x1*x2),
                 cut(45 + x1/2. + abs(x1)*x2/2. + (f[1])*x1*x2),
                 x1 + x2*abs(x1),
                 cut(45 - x1/2. - abs(x1)*x2/2. + (f[2])*x1*x2),
                 cut(45 + x1/2. + abs(x1)*x2/2. + (f[3])*x1*x2)
                 ]
        return alpha

# %%
    LAWS = {
        'law_general': alpha1,
            }

    for key in LAWS:
        alpha = LAWS[key]

        RESULT_DX = np.zeros((len(X2), len(X1)))
        RESULT_DY = np.zeros((len(X2), len(X1)))
        RESULT_DEPS = np.zeros((len(X2), len(X1)))
        RESULT_STRESS = np.zeros((len(X2), len(X1)))
        X_idx = np.zeros((len(X2), len(X1)))
        Y_idx = np.zeros((len(X2), len(X1)))
        GAITS = []

        for x1_idx, x1 in enumerate(X1):
            for x2_idx, x2 in enumerate(X2):
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
                    ref2 += + [ref2[0]]

                init_pose = pf.GeckoBotPose(
                        *model.set_initial_pose(ref2[0][0], eps,
                                                (x2_idx*dx, -x1_idx*dy)))
                gait = pf.predict_gait(ref2, init_pose, weight)

                (dxx, dyy), deps = gait.get_travel_distance()
                RESULT_DX[x2_idx][x1_idx] = dxx
                RESULT_DY[x2_idx][x1_idx] = dyy
                RESULT_DEPS[x2_idx][x1_idx] = deps
                print('(x2, x1):', round(x2, 1), round(x1, 1), ':',
                      round(deps, 2))

                Phi = gait.plot_phi()
                plt.title('Phi')

                cumstress = gait.plot_stress()
                RESULT_STRESS[x2_idx][x1_idx] = cumstress
                plt.title('Inner Stress')

                Alp = gait.plot_alpha()
                plt.title('Alpha')

                plt.figure('GeckoBotGait')

                X_idx[x2_idx][x1_idx] = x2_idx*dx
                Y_idx[x2_idx][x1_idx] = -x1_idx*dy
                GAITS.append(gait)

    # %% Save all the figs as png

        fig = plt.figure('GeckoBotGait')
        levels = [0, 25, 50, 100, 150, 200, 1300]
        contour = plt.contourf(X_idx, Y_idx, RESULT_STRESS, alpha=1,
                               cmap='OrRd', levels=levels)
        for gait in GAITS:
            gait.plot_gait()
            gait.plot_orientation()

        for xidx, x in enumerate(list(RESULT_DEPS)):
            for yidx, deps in enumerate(list(x)):
                plt.text(X_idx[xidx][yidx], Y_idx[xidx][yidx]-2.2, round(deps, 1),
                         ha="center", va="bottom",
                         bbox=dict(boxstyle="square",
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8),
                           ))

        plt.colorbar(shrink=0.5, aspect=10,
                     label="cumulative stress over gait")
        plt.clim(0, 200)

        plt.xticks(X_idx.T[0], [round(x, 1) for x in X2])
        plt.yticks(Y_idx[0], [round(x, 1) for x in X1])
        plt.xlabel('steering $x_2$')
        plt.ylabel('step length $x_1$')
        plt.axis('scaled')

        plt.grid()
        ax = fig.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    #    plt.axis('off')    
    #    fig.set_size_inches(18.5, 10.5)
        fig.set_size_inches(10.5, 8)
        fig.savefig('Out/analytic_model6/gait_{}.png'.format(key),
                    transparent=True,
                    dpi=300, bbox_inches='tight',)
# %%
        fig.clear()  # clean figure

        # %% STRUCTURE OF FIT

        X1__, X2__ = np.meshgrid(X1, X2)
        X1_ = X1__.flatten()
        X2_ = X2__.flatten()

        order = 5

        Adic = {}
        Adic[0] = [X1_*0+1]
        Adic[1] = [X1_, X2_]
        Adic[2] = [X1_**2, X2_**2, X1_*X2_]
        Adic[3] = [X1_**3, X2_**3, X1_**2*X2_, X2_**2*X1_]
        Adic[4] = [X1_**4, X2_**4, X1_**3*X2_**1, X1_**2*X2_**2, X1_**1*X2_**3]
        Adic[5] = [X1_**5, X2_**5, X1_**4*X2_**1, X1_**3*X2_**2, X1_**2*X2_**3, X1_**1*X2_**4]

        Tdic = {}
        Tdic[0] = '{}'
        Tdic[1] = ' + {}x_1^1 + {}x_2^1'
        Tdic[2] = ' + {}x_1^2 + {}x_2^2 + {}x_1^1x_2^1'
        Tdic[3] = ' + {}x_1^3 + {}x_2^3 + {}x_1^2x_2^1 + {}x_1^1x_2^2'
        Tdic[4] = ' + {}x_1^4 + {}x_2^4 + {}x_1^3x_2^1 + {}x_1^2x_2^2 + {}x_1^1x_2^3'
        Tdic[5] = ' + {}x_1^5 + {}x_2^5 + {}x_1^4x_2^1 + {}x_1^3x_2^2 + {}x_1^2x_2^3 + {}x_1^1x_2^4'

        pys = (
                '{}'
                + '+ {}*x1 + {}*x2'
                + '+ {}*x1**2 + {}*x2**2 + {}*x1**1*x2**1'
                + '+ {}*x1**3 + {}*x2**3 + {}*x1**2*x2**1 + {}*x1**1*x2**2'
                + '+ {}*x1**4 + {}*x2**4 + {}*x1**3*x2**1 + {}*x1**2*x2**2 + {}*x1**1*x2**3'
#                + '+ {}*x1**5 + {}*x2**5 + {}*x1**4*x2**1 + {}*x1**3*x2**2 + {}*x1**2*x2**3 + {}*x1**1*x2**4'
                )


        tex = ''
        for i in range(order+1):
            tex += Tdic[i]
        def flat_list(l):
            return [item for sublist in l for item in sublist]

        A = [Adic[i] for i in range(order+1)]
        A = flat_list(A)
        A = np.array(A).T

## %% Plot DEPS
        # Plot the surface.
        levels = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
        fig = plt.figure('Delta Epsilon')
        surf = plt.contourf(X1__, X2__, RESULT_DEPS, cmap=cm.coolwarm, levels=levels)
        surf = plt.contour(X1__, X2__, RESULT_DEPS, levels=levels, cmap=cm.coolwarm)

        ax1 = plt.gca()
        ax1.clabel(surf, levels, inline=True, colors='black', fmt='%2.0f')

        # Add a color bar which maps values to colors.
        plt.clim(-100, 100)
#        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.xlabel('step length $x_1$')
        plt.ylabel('steering $x_2$')

        # FIT DEPS

        B = RESULT_DEPS.flatten()
        coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=-1)
        FIT = X1_*0
        for c, a in zip(coeff, A.T):
            FIT += c*a
        FIT = np.reshape(FIT, np.shape(X1__), order='C')
        surf = plt.contour(X1__, X2__, FIT, '--', levels=levels, colors='k')

        coeff_ = [round(c, 2) for c in coeff]
        print(coeff_)
        fig = plt.gcf()
        deps = tex.format(*coeff_)
        plt.title('$\\delta \\varepsilon='+deps+'$')
        fig.set_size_inches(10.5, 8.5)
        fig.savefig('Out/analytic_model6/FitDeps_{}_order_{}.png'.format(key, order), dpi=300)

        ## %% Plot DXDY
        fig, ax = plt.subplots(num='DXDY')
        M = np.hypot(RESULT_DX, RESULT_DY)

        roundon = 2
        BDX = RESULT_DX.flatten()
        coeff, r, rank, s = np.linalg.lstsq(A, BDX)
        coeff_ = [round(c, roundon) for c in coeff]

        FITDX = X1_*0
        for c, a in zip(coeff_, A.T):
            FITDX += c*a

        dx = tex.format(*coeff_)

        BDY = RESULT_DY.flatten()
        coeff, r, rank, s = np.linalg.lstsq(A, BDY)
        coeff_ = [round(c, roundon) for c in coeff]
        dy = tex.format(*coeff_)
        
        FITDY = X1_*0
        for c, a in zip(coeff_, A.T):
            FITDY += c*a

        BDX_, BDY_ = np.around(BDX, 3), np.around(BDY, 3)
        FITDX_, FITDY_ = np.around(FITDX, 3), np.around(FITDY, 3)
        error_x = np.reshape(((BDX_ - FITDX_)), np.shape(X1__), order='C')
        error_y = np.reshape(((BDY_ - FITDY_)), np.shape(X1__), order='C')
        error_len_abs = np.sqrt((error_x**2 + error_y**2))
        error_len_rel = error_len_abs / np.reshape(np.sqrt(BDX_**2 + BDY_**2), np.shape(X1__)) * 100
#        error_len_rel[error_len_rel == np.inf] = 0
#        error_len_rel[np.isnan(error_len_rel)] = 0
        levels = [0, 5, 10, 15, 20, 25, 30, 50]
        contour = plt.contourf(X1__, X2__, error_len_rel, cmap='OrRd',
                              levels=levels)
        plt.colorbar(label='$|FIT - SIM|/|SIM|$')
#        plt.contour(contour, levels=levels)  #, colors='k')
#        plt.clabel(contour, levels, inline=True)  #, colors='k')
        plt.clim(0, 30)

        # PLOT VECTOR FIELD
        q = ax.quiver(X1__, X2__, RESULT_DX, RESULT_DY, M, units='x', scale=.2)
        ax.scatter(X1__, X2__, color='0.5', s=10)
        ax.quiver(X1__, X2__, FITDX_, FITDY_, units='x', scale=.2, color='black')
        ax.quiver(X1__, X2__, error_x, error_y, units='x', scale=.2, color='red')
        ax.grid()

        print('dx =' + dx)
        print('dy =' + dy)

        tit = '$\delta x(x_1, x_2)= \\begin{array}{c} %s \\\ %s \\end{array}$' % (dx, dy)
        plt.title(tit)

        fig = plt.gcf()
        fig.set_size_inches(10.5, 8.5)
        fig.savefig('Out/analytic_model6/FitDXDY_{}_order_{}.png'.format(key, order), dpi=300)
    
        # %%
        
    
    plt.show()
