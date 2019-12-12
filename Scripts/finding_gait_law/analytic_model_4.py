#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 01:25:51 2019

@author: ls
"""

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
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

    X1 = np.arange(70.01, 90.2, 10.)
    X2 = np.arange(-.5, .52, .2)

    n_cyc = 1

    dx, dy = 3.5, 3+2.5*n_cyc

    def cut(x):
        return x if x > 0.001 else 0.001

    def alpha_original(x1, x2, f):
        alpha = [cut(45 - x1/2. + (f[0])*x1*x2 + (f[0] ^ 1)*abs(x1)*x2/2.),
                 cut(45 + x1/2. + (f[1])*x1*x2 + (f[1] ^ 1)*abs(x1)*x2/2.),
                 x1 + x2*abs(x1),
                 cut(45 - x1/2. + (f[2])*x1*x2 + (f[2] ^ 1)*abs(x1)*x2/2.),
                 cut(45 + x1/2. + (f[3])*x1*x2 + (f[3] ^ 1)*abs(x1)*x2/2.)
                 ]
        return alpha

    def alpha1(x1, x2, f):
        alpha = [cut(45 - x1/2. - abs(x1)*x2/2. + (f[0])*x1*x2),
                 cut(45 + x1/2. + abs(x1)*x2/2. + (f[1])*x1*x2),
                 x1 + x2*abs(x1),
                 cut(45 - x1/2. - abs(x1)*x2/2. + (f[2])*x1*x2),
                 cut(45 + x1/2. + abs(x1)*x2/2. + (f[3])*x1*x2)
                 ]
        return alpha

    def alpha1_(x1, x2, f):
        alpha = [cut(45 - x1/2. - abs(x1)*x2/2. + (f[0])*x1*x2/2 - (f[0] ^ 1)*x1*x2/2),
                 cut(45 + x1/2. + abs(x1)*x2/2. + (f[1])*x1*x2/2 - (f[1] ^ 1)*x1*x2/2),
                 x1 + x2*abs(x1),
                 cut(45 - x1/2. - abs(x1)*x2/2. + (f[2])*x1*x2/2 - (f[2] ^ 1)*x1*x2/2),
                 cut(45 + x1/2. + abs(x1)*x2/2. + (f[3])*x1*x2/2 - (f[3] ^ 1)*x1*x2/2)
                 ]
        return alpha

    def alpha2(x1, x2, f):
        alpha = [cut(45 - x1/2. - abs(x1)*x2/2. + (f[0])*x1*x2),
                 cut(45 + x1/2. + abs(x1)*x2/2. - (f[1])*x1*x2),
                 x1 + x2*abs(x1),
                 cut(45 - x1/2. - abs(x1)*x2/2. + (f[2])*x1*x2),
                 cut(45 + x1/2. + abs(x1)*x2/2. - (f[3])*x1*x2)
                 ]
        return alpha

    def alpha3(x1, x2, f):
        alpha = [cut(45 - x1/2. - abs(x1)*x2/2. - (f[0])*x1*x2),
                 cut(45 + x1/2. + abs(x1)*x2/2. + (f[1])*x1*x2),
                 x1 + x2*abs(x1),
                 cut(45 - x1/2. - abs(x1)*x2/2. - (f[2])*x1*x2),
                 cut(45 + x1/2. + abs(x1)*x2/2. + (f[3])*x1*x2)
                 ]
        return alpha

    def alpha4(x1, x2, f):
        alpha = [cut(45 - x1/2. - abs(x1)*x2/2. - (f[0])*x1*x2),
                 cut(45 + x1/2. + abs(x1)*x2/2. - (f[1])*x1*x2),
                 x1 + x2*abs(x1),
                 cut(45 - x1/2. - abs(x1)*x2/2. - (f[2])*x1*x2),
                 cut(45 + x1/2. + abs(x1)*x2/2. - (f[3])*x1*x2)
                 ]
        return alpha


# %%
    LAWS = {
#        'law_original': alpha_original,
#        'law0': alpha1,
#        'law0a': alpha1_,
        'law1': alpha2,
#        'law2': alpha3,
#        'law3': alpha4,
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
    #            if x1_idx == 0:
    #                plt.figure('GeckoBotGait')
    #                plt.text(x2_idx*dx, 1+n_cyc, 'x2={}'.format(round(x2, 1)))
                f1 = [0, 1, 1, 0]
                f2 = [1, 0, 0, 1]
#                if x2 < 0:
                if 1:
                    ref2 = [[alpha(-x1, x2, f2), f2],
                            [alpha(x1, x2, f1), f1]
                            ]
#                else:
#                    ref2 = [[alpha(x1, x2, f1), f1],
#                            [alpha(-x1, x2, f2), f2]
#                            ]
                ref2 = ref2*n_cyc + [ref2[0]]
    
                init_pose = pf.GeckoBotPose(
                        *model.set_initial_pose(ref2[0][0], eps,
                                                (x2_idx*dx, -x1_idx*dy)))
                gait = pf.predict_gait(ref2, init_pose, weight)
    
                (dxx, dyy), deps = gait.get_travel_distance()
                RESULT_DX[x2_idx][x1_idx] = dxx
                RESULT_DY[x2_idx][x1_idx] = dyy
                RESULT_DEPS[x2_idx][x1_idx] = deps
                print('(x2, x1):', round(x2, 1), round(x1, 1), ':', round(deps, 2))
    
                Phi = gait.plot_phi()
                plt.title('Phi')
    
                cumstress = gait.plot_stress()
                RESULT_STRESS[x2_idx][x1_idx] = cumstress
                plt.title('Inner Stress')
    
                Alp = gait.plot_alpha()
                plt.title('Alpha')
    
                plt.figure('GeckoBotGait')
                
    #            plt.text(x2_idx*dx, -x1_idx*dy - 3, '{}'.format(round(cumstress, 1)))
                X_idx[x2_idx][x1_idx] = x2_idx*dx
                Y_idx[x2_idx][x1_idx] = -x1_idx*dy
                GAITS.append(gait)
    
    #        plt.figure('GeckoBotGait')
    #        plt.text(-4.5, -x1_idx*dy, 'x1={}'.format(int(x1)))
    
    
    # %% Save all the figs as png
    
        fig = plt.figure('GeckoBotGait')
        levels = [0, 25, 50, 100, 150, 200, 1300]
        contour = plt.contourf(X_idx, Y_idx, RESULT_STRESS, alpha=1, cmap='OrRd',
                               levels=levels)
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
    
        plt.colorbar(shrink=0.5, aspect=10, label="cumulative stress over gait")
        plt.clim(0,200)
        
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
        fig.savefig('Out/analytic_model4/gait_{}.png'.format(key), transparent=False,
                    dpi=300, bbox_inches='tight',)
# %%
        fig.clear()  # clean figure
    # %%
    
#        plt.figure('GeckoBotGaitAlphaHistory')
#        fig = plt.gcf()
#        fig.set_size_inches(18.5, 10.5)
#        fig.savefig('Out/analytic_model4/GeckoBotGaitAlphaHistory_{}.png'.format(aidx), dpi=300)
#    
#        plt.figure('GeckoBotGaitPhiHistory')
#        fig = plt.gcf()
#        fig.set_size_inches(18.5, 10.5)
#        fig.savefig('Out/analytic_model4/GeckoBotGaitPhiHistory_{}.png'.format(aidx), dpi=300)
#    
#        plt.figure('GeckoBotGaitStress')
#        fig = plt.gcf()
#        fig.set_size_inches(18.5, 10.5)
#        fig.savefig('Out/analytic_model4/GeckoBotGaitStress_{}.png'.format(aidx), dpi=300)
    
        # %% Plot DEPS
    
#        X1__, X2__ = np.meshgrid(X1, X2)
#        # Plot the surface.
#        fig = plt.figure('Delta Epsilon')
#        ax = fig.gca(projection='3d')
#        surf = ax.plot_surface(X1__, X2__, RESULT_DEPS, cmap=cm.coolwarm,
#                               linewidth=0, antialiased=False)
#        # Add a color bar which maps values to colors.
#        fig.colorbar(surf, shrink=0.5, aspect=5)
#        plt.xlabel('x1')
#        plt.ylabel('x2')
#        plt.title('Delta Epsilon')
    
        # %% FIT DEPS
#        X1_ = X1__.flatten()
#        X2_ = X2__.flatten()
#        # deps(gam, x) = c0 + c1*x1 + c2*x2 + c3*x1**2 + c4*x2**2 + c5*x1*x2
#        A = np.array([X1_*0+1, X1_, X2_, X1_**2, X2_**2, X1_*X2_]).T
#    
#        B = RESULT_DEPS.flatten()
#        coeff, r, rank, s = np.linalg.lstsq(A, B)
#        FIT = (coeff[0] + coeff[1]*X1__ + coeff[2]*X2__ + coeff[3]*X1__**2
#               + coeff[4]*X2__**2 + coeff[5]*X1__*X2__)
#        surf = ax.plot_wireframe(X1__, X2__, FIT, rcount=10, ccount=10)
#    
#        coeff_ = [round(c, 2) for c in coeff]
#        plt.title('deps(x1, x2) = {} + {}*x1 + {}*x2 + {}*x1^2 + {}*x2**2 + {}*x1*x2'.format(*coeff_))
#        fig = plt.gcf()
#        fig.set_size_inches(18.5, 10.5)
#        fig.savefig('Out/analytic_model4/FitDeps_{}.png'.format(aidx), dpi=300)
    
        # %% Plot DXDY
#        fig, ax = plt.subplots(num='DXDY')
#        M = np.hypot(RESULT_DX, RESULT_DY)
#        q = ax.quiver(X1__, X2__, RESULT_DX, RESULT_DY, M, units='x', scale=.2)
#        
#    #    q = ax.quiver(X1__, X2__, RESULT_DX, RESULT_DY, M, units='x', scale=2)
#        ax.scatter(X1__, X2__, color='0.5', s=10)
#    
#        ## %% FIT DX DY
#        X1_ = X1__.flatten()
#        X2_ = X2__.flatten()
#        # deps(gam, x) = c0 + c1*x1 + c2*x2 + c3*x1**2 + c4*x2**2 + c5*x1*x2
#        A = np.array([X1_*0+1, X1_, X2_, X1_**2, X2_**2, X1_*X2_]).T
#    
#        BDX = RESULT_DX.flatten()
#        coeff, r, rank, s = np.linalg.lstsq(A, BDX)
#        FITDX = (coeff[0] + coeff[1]*X1__ + coeff[2]*X2__ + coeff[3]*X1__**2
#                 + coeff[4]*X2__**2 + coeff[5]*X1__*X2__)
#        coeff_ = [round(c, 2) for c in coeff]
#        dx = '{} + {}x_1 + {}x_2 + {}x_1^2 + {}x_2^2 + {}x_1x_2'.format(*coeff_)
#    
#        BDY = RESULT_DY.flatten()
#        coeff, r, rank, s = np.linalg.lstsq(A, BDY)
#        coeff_ = [round(c, 2) for c in coeff]
#        dy = '{} + {}x_1 + {}x_2 + {}x_1^2 + {}x_2^2 + {}x_1x_2'.format(*coeff_)
#        FITDY = (coeff[0] + coeff[1]*X1__ + coeff[2]*X2__ + coeff[3]*X1__**2
#                 + coeff[4]*X2__**2 + coeff[5]*X1__*X2__)
#    
#        ax.quiver(X1__, X2__, FITDX, FITDY, units='x', scale=.2)
#        ax.grid()
#    
#        print('dx =' + dx)
#        print('dy =' + dy)
#    #    rc('text', usetex=True)
#    #    tit = '$\delta x(x_1, x_2)= \\begin{array}{c} %s \\\ %s \\end{array}$' % (dx, dy)
#        tit = 'Leider kein TeX'
#        plt.title(tit)
#    
#        fig = plt.gcf()
#        fig.set_size_inches(18.5, 10.5)
#        fig.savefig('Out/analytic_model4/FitDXDY_{}.png'.format(aidx), dpi=300)
    
        # %%
        
    
    plt.show()
