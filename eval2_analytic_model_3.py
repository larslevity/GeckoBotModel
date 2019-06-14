# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:03:39 2019

@author: AmP
"""

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    from Src.Utils import plot_fun as pf
    from Src.Math import kinematic_model as model


    eps = 90
    f_l, f_o, f_a = 10, 1, 10
    weight = [f_l, f_o, f_a]

    X1 = np.arange(60.01, 90.2, 9.9)
    X2 = np.arange(-.51, .52, .1)

    RESULT_DX = np.zeros((len(X2), len(X1)))
    RESULT_DY = np.zeros((len(X2), len(X1)))
    RESULT_DEPS = np.zeros((len(X2), len(X1)))

    n_cyc = 2

    dx, dy = 3.5, 3+2.5*n_cyc

    def cut(x):
        return x if x > 0.001 else 0.001

    def x4fun(x1, x2):
        return 0 + 2*abs(x1*x2) - 80*abs(x2)

    def x3fun(x1, x2):
        return 30 - 2*abs(x1*x2) + 110*abs(x2)

    def alpha(x1, x2, f):
        x3 = x3fun(x1, x2)
        x4 = x4fun(x1, x2)
#        alpha = [45 - x1/2. + (f[0] ^ 1)*(x4) + f[0]*x3,
#                 45 + x1/2. + (f[1] ^ 1)*(x4) + f[1]*x3,
#                 x1 + x2*abs(x1),
#                 45 - x1/2. + (f[2] ^ 1)*(x3) + f[2]*x4,
#                 45 + x1/2. + (f[3] ^ 1)*(x3) + f[3]*x4
#                 ]
        alpha = [45 - x1/2. + (f[0] ^ 1)*(x3) + f[0]*x4,
                 45 + x1/2. + (f[1] ^ 1)*(x3) + f[1]*x4,
                 x1 + x2*abs(x1),
                 45 - x1/2. + (f[2] ^ 1)*(x4) + f[2]*x3,
                 45 + x1/2. + (f[3] ^ 1)*(x4) + f[3]*x3
                 ]
        return alpha

    for x1_idx, x1 in enumerate(X1):
        for x2_idx, x2 in enumerate(X2):
            if x1_idx == 0:
                plt.figure('GeckoBotGait')
                plt.text(x2_idx*dx, 1+n_cyc, 'x1={}'.format(round(x2, 2)))
            f1 = [0, 1, 1, 0]
            f2 = [1, 0, 0, 1]
            if x2 < 0:
                ref2 = [[alpha(-x1, x2, f2), f2],
#                        [alpha(x1, -x2, f2), f1],
                        [alpha(x1, x2, f1), f1],
#                        [alpha(x1, x2, f1), f2]
                        ]
            else:
                ref2 = [[alpha(x1, x2, f1), f1],
#                        [alpha(x1, x2, f1), f2],
                        [alpha(-x1, x2, f2), f2],
#                        [alpha(x1, -x2, f2), f1]
                        ]
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

            gait.plot_gait()
            plt.title('Gait')
            gait.plot_orientation()

            Phi = gait.plot_phi()
            plt.title('Phi')

            cumstress = gait.plot_stress()
            plt.title('Inner Stress')

            Alp = gait.plot_alpha()
            plt.title('Alpha')

            plt.figure('GeckoBotGait')
            plt.text(x2_idx*dx, -x1_idx*dy - 2.5, '{}'.format(round(deps, 1)))
            plt.text(x2_idx*dx, -x1_idx*dy - 3, '{}'.format(round(cumstress, 1)))

        plt.figure('GeckoBotGait')
        plt.text(-4.5, -x1_idx*dy, 'x2={}'.format(round(x1, 0)))


# %% Save all the figs as png

    plt.figure('GeckoBotGait')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('Out/analytic_model3/gait.png', transparent=False,
                dpi=300)

    plt.figure('GeckoBotGaitAlphaHistory')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('Out/analytic_model3/GeckoBotGaitAlphaHistory.png', dpi=300)

    plt.figure('GeckoBotGaitPhiHistory')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('Out/analytic_model3/GeckoBotGaitPhiHistory.png', dpi=300)

    plt.figure('GeckoBotGaitStress')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('Out/analytic_model3/GeckoBotGaitStress.png', dpi=300)

    # %%

    X1__, X2__ = np.meshgrid(X1, X2)
    # Plot the surface.
    fig = plt.figure('Delta Epsilon')
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X1__, X2__, RESULT_DEPS, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Delta Epsilon')

    X1_ = X1__.flatten()
    X2_ = X2__.flatten()
    # deps(gam, x) = c0 + c1*gam + c2*x + c3*gam**2 + c4*x**2 + c5*gam*x
    A = np.array([X1_*0+1, X1_, X2_, X1_**2, X2_**2, X1_*X2_]).T
#    # deps(gam, x) = c0 + c1*gam + c2*x
#    A = np.array([X*0+1, X, Y]).T

    B = RESULT_DEPS.flatten()
    coeff, r, rank, s = np.linalg.lstsq(A, B)
    FIT = (coeff[0] + coeff[1]*X1__ + coeff[2]*X2__ + coeff[3]*X1__**2
           + coeff[4]*X2__**2 + coeff[5]*X1__*X2__)
    surf = ax.plot_wireframe(X1__, X2__, FIT, rcount=10, ccount=10)

    coeff_ = [round(c, 2) for c in coeff]
    plt.title('deps(x1, x2) = {} + {}*x1 + {}*x2 + {}*x1^2 + {}*x2**2 + {}*x1*x2'.format(*coeff_))



    plt.show()


