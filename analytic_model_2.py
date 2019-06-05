# -*- coding: utf-8 -*-
"""
Created on Mon Jun 03 09:12:03 2019

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

    X2 = np.arange(80.01, 90.2, 9.9)
    X1 = np.arange(.11, .52, .1)

    RESULT_DX = np.zeros((len(X1), len(X2)))
    RESULT_DY = np.zeros((len(X1), len(X2)))
    RESULT_DEPS = np.zeros((len(X1), len(X2)))

    n_cyc = 2

    dx, dy = 3.5, 1+2.5*n_cyc

    def alpha(x1, x2, f):
        alpha = [45 - x2/2. + (f[0] ^ 1)*abs(x1*x2) + f[0]*abs(x2)*x1,
                 45 + x2/2. + (f[1] ^ 1)*abs(x1*x2) + f[1]*abs(x2)*x1,
                 x2 + x1*abs(x2),
                 45 - x2/2. + (f[2] ^ 1)*abs(x1*x2) + f[2]*abs(x2)*x1,
                 45 + x2/2. + (f[3] ^ 1)*abs(x1*x2) + f[3]*abs(x2)*x1
                 ]
        return alpha

    for x2_idx, x2 in enumerate(X2):
        for x1_idx, x1 in enumerate(X1):
            if x2_idx == 0:
                plt.figure('GeckoBotGait')
                plt.text(x1_idx*dx, 1+n_cyc, 'x1={}'.format(round(x1, 2)))
            f1 = [0, 1, 1, 0]
            f2 = [1, 0, 0, 1]
            if x1 < 0:
                ref2 = [[alpha(x1, -x2, f2), f2],
#                        [alpha(x1, -x2, f2), f1],
                        [alpha(x1, x2, f1), f1],
#                        [alpha(x1, x2, f1), f2]
                        ]
            else:
                ref2 = [[alpha(x1, x2, f1), f1],
#                        [alpha(x1, x2, f1), f2],
                        [alpha(x1, -x2, f2), f2],
#                        [alpha(x1, -x2, f2), f1]
                        ]
            ref2 = ref2*n_cyc + [ref2[0]]

            init_pose = pf.GeckoBotPose(
                    *model.set_initial_pose(ref2[0][0], eps,
                                            (x1_idx*dx, -x2_idx*dy)))
            gait = pf.predict_gait(ref2, init_pose, weight)

            (dxx, dyy), deps = gait.get_travel_distance()
            RESULT_DX[x1_idx][x2_idx] = dxx
            RESULT_DY[x1_idx][x2_idx] = dyy
            RESULT_DEPS[x1_idx][x2_idx] = deps
            print('(x2, x1):', round(x2, 1), round(x1, 1), ':', round(deps, 2))

            plt.figure('GeckoBotGait')
            plt.text(x1_idx*dx, -x2_idx*dy - 2.5, '{}'.format(round(deps, 1)))

            gait.plot_gait()
            gait.plot_orientation()
            Phi = gait.plot_phi()
            gait.plot_stress()
            Alp = gait.plot_alpha()

        plt.figure('GeckoBotGait')
        plt.text(-4.5, -x2_idx*dy, 'x2={}'.format(round(x2, 0)))

    # %%

    X2_, X1_ = np.meshgrid(X2, X1)
    # Plot the surface.
    fig = plt.figure('Delta Epsilon')
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X2_, X1_, RESULT_DEPS, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('x2')
    plt.ylabel('x1')
    plt.title('Delta Epsilon')

    X = X2_.flatten()
    Y = X1_.flatten()
    # deps(gam, x) = c0 + c1*gam + c2*x + c3*gam**2 + c4*x**2 + c5*gam*x
    A = np.array([X*0+1, X, Y, X**2, Y**2, X*Y]).T
#    # deps(gam, x) = c0 + c1*gam + c2*x
#    A = np.array([X*0+1, X, Y]).T

    B = RESULT_DEPS.flatten()
    coeff, r, rank, s = np.linalg.lstsq(A, B)
    FIT = (coeff[0] + coeff[1]*X2_ + coeff[2]*X1_ + coeff[3]*X2_**2
           + coeff[4]*X1_**2 + coeff[5]*X2_*X1_)
    surf = ax.plot_wireframe(X2_, X1_, FIT, rcount=10, ccount=10)

    coeff_ = [round(c, 2) for c in coeff]
    plt.title('deps(x1, x2) = {} + {}*x2 + {}*x1 + {}*x2^2 + {}*x1**2 + {}*x2*x1'.format(*coeff_))

#    # Plot the surface.
#    fig = plt.figure('Delta x')
#    ax = fig.gca(projection='3d')
#    surf = ax.plot_surface(X2_, X1_, RESULT_DX, cmap=cm.coolwarm,
#                           linewidth=0, antialiased=False)
#    # Add a color bar which maps values to colors.
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#    plt.xlabel('x2')
#    plt.ylabel('x1')
#    plt.title('Delta x')
#
#    # Plot the surface.
#    fig = plt.figure('Delta y')
#    ax = fig.gca(projection='3d')
#    surf = ax.plot_surface(X2_, X1_, RESULT_DY, cmap=cm.coolwarm,
#                           linewidth=0, antialiased=False)
#    # Add a color bar which maps values to colors.
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#    plt.xlabel('x2')
#    plt.ylabel('x1')
#    plt.title('Delta y')

    plt.show()

    # %% Auswertung
    # Model:    alp0,4 = 45 + x2/2 + not(f[0])*abs(x1)
    #           alp1,3 = 45 - x2/2 + not(f[0])*abs(x1)
    #           alp2   = x2 + x1
    # deps (c1 + c2*x2 + c3*x1 + c4*x1*x2)

    A = np.matrix([
            [-0.02,     .00,    -.32,    -.0],
            [0.040,     .00,    -.38,    -.01],
            [-1.36,     .05,    -.43,    -.03]
            ])
    cycs = np.diag([1/2., 1/5., 1/10.])
    check = cycs*A
    print(check)

