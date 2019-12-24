# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 09:22:41 2019

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
    from Src.Utils import save as save_utils
    from Src.Math import kinematic_model as model

    from matplotlib import rc
    rc('text', usetex=True)

    res = 15  # resolution of gait
#    startposes = ['mirror', 'normal']  # 'normal' / 'mirror'
    startposes = ['normal']  # 'normal' / 'mirror'

    eps0 = 90
    c1 = .5
    len_leg, len_tor = [9.1, 10.3]
    
    f_l, f_o, f_a = .2, 12.1, 6.1
#    f_l, f_o, f_a = .1, 1, 10
    
    weight = [f_l, f_o, f_a]
    

    X1 = list(np.linspace(-90, 90, res))
#    X2 = [-.5, .5, -.2, .2]
    X2 = [-.25]
    TIME = np.linspace(0, 1, 2*res+1)

    def cut(x):
        return x if x > 0.001 else 0.001


    def alpha1(x1, x2, f, c1=1):
        alpha = [cut(45 - x1/2. - abs(x1)*x2/2. + (f[0])*x1*x2*c1 + (f[0]^1)*x1*x2*c1),
                 cut(45 + x1/2. + abs(x1)*x2/2. + (f[1])*x1*x2*c1 + (f[1]^1)*x1*x2*c1),
                 x1 + x2*abs(x1),
                 cut(45 - x1/2. - abs(x1)*x2/2. + (f[2])*x1*x2*c1 + (f[2]^1)*x1*x2*c1),
                 cut(45 + x1/2. + abs(x1)*x2/2. + (f[3])*x1*x2*c1 + (f[3]^1)*x1*x2*c1)
                 ]
        return alpha


    def alpha2(x1, x2, f, c1=.5):
        alpha = [cut(45 - x1/2. - abs(x1)*x2/2. + x1*x2*c1 - (f[0])*x1*x2*c1/2),
                 cut(45 + x1/2. + abs(x1)*x2/2. + x1*x2*c1 - (f[1])*x1*x2*c1/2),
                 x1 + x2*abs(x1),
                 cut(45 - x1/2. - abs(x1)*x2/2. + x1*x2*c1 - (f[2])*x1*x2*c1/2),
                 cut(45 + x1/2. + abs(x1)*x2/2. + x1*x2*c1 - (f[3])*x1*x2*c1/2)
                 ]
        return alpha


    def alpha3(x1, x2, f, c1=.5):
        alpha = [cut(45 - x1/2. - abs(x1)*x2/2. + x1*x2*c1),
                 cut(45 + x1/2. + abs(x1)*x2/2. + x1*x2*c1),
                 x1 + x2*abs(x1),
                 cut(45 - x1/2. - abs(x1)*x2/2. + x1*x2*c1),
                 cut(45 + x1/2. + abs(x1)*x2/2. + x1*x2*c1)
                 ]
        return alpha



# %%
    LAWS = {
        'law0': alpha3,
            }
    for startpose in startposes:
        for key in LAWS:
            alpha = LAWS[key]

            for x2_idx, x2 in enumerate(X2):
                RESULT = {'alp': [],
                          'eps': [],
                          'phi': []}

                REF = {'alp': [[], [], [], [], []],
                       'eps': [[]],
                       'phi': [[], [], [], []],
                       'f1': [],
                       'f2': []}

                GAITS = []

                f1 = [0, 1, 1, 0]
                f2 = [1, 0, 0, 1]
                x10 = X1[0]
                ref2 = []

                if startpose == 'normal':
                    # init pose:
                    alp = alpha(X1[0], x2, f1)
                    for idx, alpi in enumerate(alp):
                        REF['alp'][idx].append(alpi)
                    REF['eps'][0].append(0)
                    REF['f1'].append(1)
                    REF['f2'].append(0)
                    for x1 in X1:
                        alp = alpha(x1, x2, f1, c1)
                        ref2.append([alp, f1])
                        for idx, alpi in enumerate(alp):
                            REF['alp'][idx].append(alpi)
                        REF['eps'][0].append((X1[0]*x2*c1 -x1*x2*c1)/1)
                        REF['f1'].append(1)
                        REF['f2'].append(0)
                    for x1 in X1[::-1]:  # reverse
                        alp = alpha(x1, x2, f2, c1)
                        ref2.append([alp, f2])
                        for idx, alpi in enumerate(alp):
                            REF['alp'][idx].append(alpi)
                        REF['eps'][0].append((3*c1*X1[0]*x2 + x1*x2*c1)/1)
                        REF['f1'].append(0)
                        REF['f2'].append(1)

                else:
                    for x1 in X1:
                        ref2.append([alpha(-x1, x2, f2), f2])


        

                init_pose = pf.GeckoBotPose(
                        *model.set_initial_pose(ref2[0][0], eps0, (0, 0),
                                                len_leg=len_leg, len_tor=len_tor))
                gait = pf.predict_gait(ref2, init_pose, weight, (len_leg, len_tor))

                (dxx, dyy), deps = gait.get_travel_distance()
                print('(x2, x1):', round(x2, 1), round(x1, 1), ':', round(deps, 2))

                Phi = gait.plot_phi()
                for phi in Phi:
                    RESULT['phi'].append(phi)
                plt.title('Phi')

                cumstress = gait.plot_stress()
                plt.title('Inner Stress')

                Alp = gait.plot_alpha()
                for alp in Alp:
                    RESULT['alp'].append(alp)
                plt.title('Alpha')

                Eps = gait.plot_epsilon()
                RESULT['eps'].append(Eps)
                plt.title('Epsilon')

                plt.figure('GeckoBotGait')
                gait.plot_gait()
                gait.plot_orientation()

                for sim_idx in range(len(RESULT['alp'][0])):
                    REF['phi'][0].append((
                        REF['eps'][0][sim_idx] - REF['alp'][2][sim_idx]/2.
                        - REF['alp'][0][sim_idx] + eps0))
                    REF['phi'][1].append((
                        REF['eps'][0][sim_idx] - REF['alp'][2][sim_idx]/2.
                        + REF['alp'][1][sim_idx] + eps0))
                    REF['phi'][2].append((
                        REF['eps'][0][sim_idx] + REF['alp'][2][sim_idx]/2.
                        + REF['alp'][3][sim_idx] + 180 + eps0))
                    REF['phi'][3].append((
                        REF['eps'][0][sim_idx] + REF['alp'][2][sim_idx]/2.
                        - REF['alp'][4][sim_idx] + 180 + eps0))

                GAITS.append(gait)

            # %% Plot Configuration space

            # PLOT ALP
                fig = plt.figure('ALP'+key+str(x2)+startpose)
                plt.plot(TIME, RESULT['alp'][2], 'orange', label='a2')
                plt.plot(TIME, REF['alp'][2], ':', color='orange')

                # for normal start:
                switch_idx = REF['f1'].index(0)
                t1 = TIME[:switch_idx]
                t2 = TIME[switch_idx-1:]

                plt.plot(t1, RESULT['alp'][0][:switch_idx], '--', color='red', label='a0')
                plt.plot(t2, RESULT['alp'][0][switch_idx-1:], '-', color='red')
                plt.plot(TIME, REF['alp'][0], ':', color='red')

                plt.plot(t1, RESULT['alp'][4][:switch_idx], '--', color='darkblue', label='a4')
                plt.plot(t2, RESULT['alp'][4][switch_idx-1:], '-', color='darkblue')
                plt.plot(TIME, REF['alp'][4], ':', color='darkblue')

                plt.plot(t1, RESULT['alp'][3][:switch_idx], '-', color='blue', label='a3')
                plt.plot(t2, RESULT['alp'][3][switch_idx-1:], '--', color='blue')
                plt.plot(TIME, REF['alp'][3], ':', color='blue')
                

                plt.plot(t1, RESULT['alp'][1][:switch_idx], '-', color='darkred', label='a1')
                plt.plot(t2, RESULT['alp'][1][switch_idx-1:], '--', color='darkred')
                plt.plot(TIME, REF['alp'][1], ':', color='darkred')


                # save
                pih = '$\\frac{\\pi}{2}$'
                plt.xticks([0, .25, .5, .75, 1], ['0', '', '0.5', '', '1'])
                plt.yticks([-90, -45, 0, 45, 90, 135],
                           ['-'+pih, '', '0', '', pih])
                plt.xlabel('time in cycle')
                plt.ylabel('bending angle')

                plt.grid()
    #            plt.legend(loc='center left', bbox_to_anchor=(1, .5))

                plt.axis('tight')
                plt.xlim([0, 1])
                plt.ylim([-135, 160.43])

                name = 'ALPHA_{}_{}_{}.tex'.format(
                        key, startpose, str(x2).replace('.', '_'))
                kwargs = {'extra_axis_parameters':
                      {'anchor=origin'}}
                save_utils.save_plt_as_tikz('Out/analytic_model8/'+name, **kwargs)
#



                
            # %% PLOT PHI / EPS
#                fig.clear()
                fig = plt.figure('PHI'+key+str(x2)+startpose)

                plt.plot(TIME, np.array(RESULT['eps'][0])-eps0, 'green', label='eps')
                plt.plot(TIME, REF['eps'][0], ':', color='green', label='eps')

# %

                plt.plot(t1, RESULT['phi'][1][:switch_idx], '-', color='darkred', label='phi1')
                plt.plot(t2, RESULT['phi'][1][switch_idx-1:], '--', color='darkred')
                plt.plot(TIME, REF['phi'][1], ':', color='darkred')
                
                plt.plot(t1, RESULT['phi'][0][:switch_idx], '--', color='red', label='phi0')
                plt.plot(t2, RESULT['phi'][0][switch_idx-1:], '-', color='red')
                plt.plot(TIME, REF['phi'][0], ':', color='red')

                plt.plot(t1, RESULT['phi'][3][:switch_idx], '--', color='darkblue', label='phi3')
                plt.plot(t2, RESULT['phi'][3][switch_idx-1:], '-', color='darkblue')
                plt.plot(TIME, REF['phi'][3], ':', color='darkblue')
# %

                plt.plot(t1, RESULT['phi'][2][:switch_idx], '-', color='blue', label='phi2')
                plt.plot(t2, RESULT['phi'][2][switch_idx-1:], '--', color='blue')
                plt.plot(TIME, REF['phi'][2], ':', color='blue')
#                
                
                plt.ylabel('phi / epsilon [deg]')
                plt.xlabel('time in cycle')


                def pifrac(a, b):
                    return '$\\frac{' + str(a) + '\\pi}{'+ str(b) + '}$'
                plt.xticks([0, .25, .5, .75, 1], ['0', '', '0.5', '', '1'])
                plt.yticks([0, 90, 180, 270, 360],
                           ['0', pifrac('', 2), '$\\pi$', pifrac(3, 2), '$2\\pi$'])
                


                plt.grid()
                kwargs = {'extra_axis_parameters':
                      {'anchor=origin'}}                
                name = 'PHI_{}_{}_{}.tex'.format(
                        key, startpose, str(x2).replace('.', '_'))
                save_utils.save_plt_as_tikz('Out/analytic_model8/'+name, **kwargs)
            # %%
#                fig.clear()
                pf.save_animation(gait.animate(), '../../Out/analytic_model8/gait.mp4')

            # %% POSES
            gait_ = pf.GeckoBotGait()
            gait_.append_pose(gait.poses[1])  # init pose
            gait_.append_pose(gait.poses[int((res))])  # midpose
            gait_.append_pose(gait.poses[int((res+1))])  # midpose
            gait_.append_pose(gait.poses[-1])  # end pose
            gait_.plot_gait()
            gait_.plot_orientation(length=5)
            gait_.save_as_tikz('gait')


# %%

    plt.show()
