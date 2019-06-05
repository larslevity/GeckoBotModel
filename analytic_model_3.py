# -*- coding: utf-8 -*-
"""
Created on Mon Jun 03 14:56:40 2019

@author: AmP
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import minimize
from scipy.optimize import basinhopping

from Src.Utils import plot_fun as pf
from Src.Utils import save_csv
from Src.Math import kinematic_model as model

eps = 90
n_cyc = 5
idx = 0

f_l, f_o, f_a = 10, 1, 10
weight = [f_l, f_o, f_a]

x1 = 60
x6 = .5


def alpha(x, f):
    x1, x2, x3, x4, x5, x6 = x
    alpha = [45 - x1/2. + (f[0] ^ 1)*(x2) + f[0]*x3,
             45 + x1/2. + (f[1] ^ 1)*(x2) + f[1]*x3,
             x1 + x6*abs(x1),
             45 - x1/2. + (f[2] ^ 1)*(x4) + f[2]*x5,
             45 + x1/2. + (f[3] ^ 1)*(x4) + f[3]*x5
             ]
    return alpha

def make_sample(x1, x6, x, x0, deps, cost):
    data = {}
    data['x1'] = [x1]
    data['x6'] = [x6]
    data['x0'] = [x0]
    data['x'] = [x]
    data['deps'] = [deps]
    data['cost'] = [cost]
    data['time'] = [time.asctime(time.localtime(time.time()))]
    return data
    

X1 = [60.01 + x*10 for x in range(5)]
X6 = [-.501+.05*x for x in range(21)]
X0 = [[10., 20.], [20., 10.]]

# %%
        
for x0 in X0:
    for x1 in X1:
        for x6 in X6:
            def objective(x):
                global idx
                global lastDeps
                x2, x3= x
                x = [x1, x2, x3, x3, x2, x6]
                x_ = [-x1, x2, x3, x3, x2, x6]
                f1 = [0, 1, 1, 0]
                f2 = [1, 0, 0, 1]
                if x[5] < 0:
                    ref2 = [[alpha(x_, f2), f2],
            #                [alpha(x_, f2), f1],
                            [alpha(x, f1), f1],
            #                [alpha(x, f1), f2]
                            ]
                else:
                    ref2 = [[alpha(x, f1), f1],
            #                [alpha(x, f1), f2],
                            [alpha(x_, f2), f2],
            #                [alpha(x_, f2), f1]
                            ]
                ref2 = ref2*n_cyc + [ref2[0]]
            
                init_pose = pf.GeckoBotPose(
                        *model.set_initial_pose(ref2[0][0], eps, (0, 0)))
                gait = pf.predict_gait(ref2, init_pose, weight=weight)
            
                (dxx, dyy), deps = gait.get_travel_distance()
                stress = [pose.cost for pose in gait.poses]
                cum_stress = sum(stress[3:])
                cum_stress = sum(stress)
        #        gait.plot_gait(str(idx).zfill(3))
        #        gait.plot_stress(str(idx).zfill(3))
        #        plt.show()
            
                print('x:', [round(xx, 2)for xx in x], ':', round(deps), round(cum_stress))
                idx += 1
                lastDeps = deps
            
                return cum_stress

            solution = minimize(objective, x0, method='COBYLA')
            print(solution.x)

# %%

            data = make_sample(x1, x6, solution.x, x0, lastDeps, solution.fun)
            save_csv.save_sample_as_csv(data, 'Out/190604_simulation_results.csv')


    # %%


#    class MyTakeStep(object):
#        def __init__(self, stepsize=10):
#            self.stepsize = stepsize
#    
#        def __call__(self, x):
#            s = self.stepsize
#            x[0] += np.random.uniform(-2.*s, 2.*s)
#            x[1:] += np.random.uniform(-s, s, x[1:].shape)
#            return x
#    
#    
#    mytakestep = MyTakeStep()
    
    
    #minimizer_kwargs = {"method": "BFGS"}
    #
    #ret = basinhopping(objective, x0, minimizer_kwargs=minimizer_kwargs,
    #                   niter=200, take_step=mytakestep)
    #print("global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun))
    
    


#    X2_, X1_ = np.meshgrid(X2, X1)
#    # Plot the surface.
#    fig = plt.figure('Delta Epsilon')
#    ax = fig.gca(projection='3d')
#    surf = ax.plot_surface(X2_, X1_, RESULT_DEPS, cmap=cm.coolwarm,
#                           linewidth=0, antialiased=False)
#    # Add a color bar which maps values to colors.
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#    plt.xlabel('x2')
#    plt.ylabel('x1')
#    plt.title('Delta Epsilon')
#
#    X = X2_.flatten()
#    Y = X1_.flatten()
#    # deps(gam, x) = c0 + c1*gam + c2*x + c3*gam**2 + c4*x**2 + c5*gam*x
#    A = np.array([X*0+1, X, Y, X**2, Y**2, X*Y]).T
##    # deps(gam, x) = c0 + c1*gam + c2*x
##    A = np.array([X*0+1, X, Y]).T
#
#    B = RESULT_DEPS.flatten()
#    coeff, r, rank, s = np.linalg.lstsq(A, B)
#    FIT = (coeff[0] + coeff[1]*X2_ + coeff[2]*X1_ + coeff[3]*X2_**2
#           + coeff[4]*X1_**2 + coeff[5]*X2_*X1_)
#    surf = ax.plot_wireframe(X2_, X1_, FIT, rcount=10, ccount=10)
#
#    coeff_ = [round(c, 2) for c in coeff]
#    plt.title('deps(x1, x2) = {} + {}*x2 + {}*x1 + {}*x2^2 + {}*x1**2 + {}*x2*x1'.format(*coeff_))

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


#    # %% Auswertung
#    # Model:    alp0,4 = 45 + x2/2 + not(f[0])*abs(x1)
#    #           alp1,3 = 45 - x2/2 + not(f[0])*abs(x1)
#    #           alp2   = x2 + x1
#    # deps (c1 + c2*x2 + c3*x1 + c4*x1*x2)
#
#    A = np.matrix([
#            [-0.02,     .00,    -.32,    -.0],
#            [0.040,     .00,    -.38,    -.01],
#            [-1.36,     .05,    -.43,    -.03]
#            ])
#    cycs = np.diag([1/2., 1/5., 1/10.])
#    check = cycs*A
#    print(check)

