# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:12:16 2019

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

    GAM = np.arange(60.01, 90.1, 9.9)
    DX = np.arange(-20.001, 20, 4.99)

    RESULT_DX = np.zeros((len(DX), len(GAM)))
    RESULT_DY = np.zeros((len(DX), len(GAM)))
    RESULT_DEPS = np.zeros((len(DX), len(GAM)))

    n_cyc = 4

    dx, dy = 3.5, 1+2.5*n_cyc

    for gam_idx, gam in enumerate(GAM):
        for x_idx, x in enumerate(DX):
            if gam_idx == 0:
                plt.figure('GeckoBotGait')
                plt.text(x_idx*dx, 1+n_cyc, 'x={}'.format(round(x)))

            ref2 = [
                     [[45-gam/2., 45+gam/2., gam+x, 45-gam/2., 45+gam/2.], [0, 1, 1, 0]],
                     [[45+gam/2., 45-gam/2., -gam+x, 45+gam/2., 45-gam/2.], [1, 0, 0, 1]]
                    ]*n_cyc
#            ref2 = model.flat_list(ref2)
            print('(gam, x):', round(gam, 1), round(x, 1))

            init_pose = pf.GeckoBotPose(
                    *model.set_initial_pose(ref2[0][0], eps, (x_idx*dx, -gam_idx*dy)))
            gait = pf.predict_gait(ref2, init_pose)

            (dxx, dyy), deps = gait.get_travel_distance()
            RESULT_DX[x_idx][gam_idx] = dxx
            RESULT_DY[x_idx][gam_idx] = dyy
            RESULT_DEPS[x_idx][gam_idx] = deps

            plt.figure('GeckoBotGait')
            plt.text(x_idx*dx, -gam_idx*dy - 2.5, '{}'.format(round(deps, 1)))

            gait.plot_gait()
            gait.plot_orientation()
            gait.plot_stress()

        plt.figure('GeckoBotGait')
        plt.text(-4.5, -gam_idx*dy, 'gam={}'.format(round(gam, 0)))


    # %% 

    GAM_, DX_ = np.meshgrid(GAM, DX)
    # Plot the surface.
    fig = plt.figure('Delta Epsilon')
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(GAM_, DX_, RESULT_DEPS, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('gam')
    plt.ylabel('x')
    plt.title('Delta Epsilon')
    
    X = GAM_.flatten()
    Y = DX_.flatten()
    # deps(gam, x) = c0 + c1*gam + c2*x + c3*gam**2 + c4*x**2 + c5*gam*x
    A = np.array([X*0+1, X, Y, X**2, Y**2, X*Y]).T
#    # deps(gam, x) = c0 + c1*gam + c2*x
#    A = np.array([X*0+1, X, Y]).T

    B = RESULT_DEPS.flatten()
    coeff, r, rank, s = np.linalg.lstsq(A, B)
    FIT = coeff[0] + coeff[1]*GAM_ + coeff[2]*DX_ + coeff[3]*GAM_**2 + coeff[4]*DX_**2 + coeff[5]*GAM_*DX_
    surf = ax.plot_wireframe(GAM_, DX_, FIT, rcount=10, ccount=10)

    coeff_ = [round(c, 2) for c in coeff]
    plt.title('deps(gam, x) = {} + {}*gam + {}*x + {}*gam^2 + {}*x**2 + {}*gam*x'.format(*coeff_))


    # Plot the surface.
    fig = plt.figure('Delta x')
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(GAM_, DX_, RESULT_DX, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('gam')
    plt.ylabel('x')
    plt.title('Delta x')

    # Plot the surface.
    fig = plt.figure('Delta y')
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(GAM_, DX_, RESULT_DY, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('gam')
    plt.ylabel('x')
    plt.title('Delta y')

    plt.show()
