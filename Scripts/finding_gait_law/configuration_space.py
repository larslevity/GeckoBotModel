# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:22:41 2019

@author: AmP
"""

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    from Src.Utils import plot_fun as pf
    from Src.Utils import save as save_utils
    from Src.Math import kinematic_model as model

    from matplotlib import rc
    rc('text', usetex=True)

    res = 30  # resolution of gait
    startposes = ['mirror', 'normal']  # 'normal' / 'mirror'

    eps = 90
    f_l, f_o, f_a = 10, 1, 10
    weight = [f_l, f_o, f_a]

    X1 = list(np.linspace(-90, 90, res))
    X2 = [-.5, .5, -.2, .2]

    def cut(x):
        return x if x > 0.001 else 0.001

    def fill_pattern(start, end, dig=10):
        alp1, feet1 = start
        alp2, feet2 = end
        pattern = []
        ALP = np.zeros((len(alp1), dig))
        for idx, (alp1i, alp2i) in enumerate(zip(alp1, alp2)):
            for j, val in enumerate(np.linspace(alp1i, alp2i, dig)):
                ALP[idx][j] = val
        for j in range(dig):
            pattern.append([[ALP[idx][j] for idx in range(len(alp1))], feet2])
        return pattern

    def flip(alpha, feet):
        feet = [f ^ 1 for f in feet]
        alp = [alpha[1], alpha[0], -alpha[2], alpha[4], alpha[3]]
        return [alp, feet]

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
        'law1': alpha2,
#        'law2': alpha3,
#        'law3': alpha4,
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
                       'phi': [[], [], [], []]}

                GAITS = []

                f1 = [0, 1, 1, 0]
                f2 = [1, 0, 0, 1]
                x10 = X1[0]
                ref2 = []

                if startpose == 'normal':
                    for x1 in X1:
                        ref2.append([alpha(x1, x2, f1), f1])
                    feet = f1
                else:
                    for x1 in X1:
                        ref2.append([alpha(-x1, x2, f2), f2])
                    feet = f2

                first = True
                for simidx, ref in enumerate(ref2):
                    doit = 2 if first else 1
                    first = False
                    for job in range(doit):
                        alp, f = ref
                        for idx, alpi in enumerate(alp):
                            REF['alp'][idx].append(alpi)
                        REF['eps'][0].append(-X1[simidx]*x2)

    #                ref2 = fill_pattern([alpha(x1, x2, f1), f1],
    #                                    [alpha(-x1, x2, f2), f2],
    #                                    dig=res)

                init_pose = pf.GeckoBotPose(
                        *model.set_initial_pose(ref2[0][0], eps, (0, 0)))
                gait = pf.predict_gait(ref2, init_pose, weight)

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
                        - REF['alp'][0][sim_idx] + 45 - 90))
                    REF['phi'][1].append((
                        REF['eps'][0][sim_idx] - REF['alp'][2][sim_idx]/2.
                        + REF['alp'][1][sim_idx] - 45 - 90))
                    REF['phi'][2].append((
                        REF['eps'][0][sim_idx] + REF['alp'][2][sim_idx]/2.
                        + REF['alp'][3][sim_idx] - 45 - 90))
                    REF['phi'][3].append((
                        REF['eps'][0][sim_idx] + REF['alp'][2][sim_idx]/2.
                        - REF['alp'][4][sim_idx]) + 45 - 90)

                GAITS.append(gait)

            # %% Plot Configuration space
                fig = plt.figure('ConfigurationSpace'+key+str(x2)+startpose)

                X1_ = [X1[0]] + X1  # + list(reversed(X1))
                X1__ = X1_
                if startpose != 'normal':
                    X1_ = list(-np.array(X1_))

                # remove offset:
                RESULT['eps_'] = []
                for idx, epsi in enumerate(RESULT['eps']):
                    offset = epsi[int(res/2)]
                    RESULT['eps_'].append(list(np.array(epsi) - offset))

                plt.plot(X1_, RESULT['alp'][2], 'orange', label='a2')
                plt.plot(X1_, REF['alp'][2], ':', color='orange')

                plt.plot(X1__, RESULT['eps_'][0], 'green', label='eps')
                plt.plot(X1__, REF['eps'][0], ':', color='green', label='eps')

                fix = 1 if feet[0] else 0
                plt.plot(X1_, RESULT['alp'][0], '-' if fix else '--',
                         color='red',
                         label='left, {}'.format('fix' if fix else 'free'))
                plt.plot(X1_, REF['alp'][0], ':', color='red')
                plt.plot(X1_, RESULT['alp'][3], '--' if fix else '-',
                         color='red',
                         label='left, {}'.format('free' if fix else 'fix'))
                plt.plot(X1_, REF['alp'][3], ':', color='red')

                fix = 1 if feet[1] else 0
                plt.plot(X1_, RESULT['alp'][1], '-' if fix else '--',
                         color='darkred',
                         label='right, {}'.format('fix' if fix else 'free'))
                plt.plot(X1_, REF['alp'][1], ':', color='darkred')
                plt.plot(X1_, RESULT['alp'][4], '--' if fix else '-',
                         color='darkred',
                         label='right, {}'.format('free' if fix else 'fix'))
                plt.plot(X1_, REF['alp'][4], ':', color='darkred')

                # remove offset:
                RESULT['phi_'] = []
                for idx, phi in enumerate(RESULT['phi']):
                    offset = phi[int(res/2)] + 90
                    RESULT['phi_'].append(list(np.array(phi) - offset))

                left = 0 if feet[0] else 2
                plt.plot(X1_, RESULT['phi_'][left], color='blue',
                         label='phi, left fix')
                plt.plot(X1_, REF['phi'][left], ':', color='blue')

                right = 1 if feet[1] else 3
                plt.plot(X1_, RESULT['phi_'][right], color='darkblue',
                         label='phi, right fix')
                plt.plot(X1_, REF['phi'][right], ':', color='darkblue')

                pih = '$\\frac{\\pi}{2}$'
                plt.xticks([-90, -45, 0, 45, 90], ['-'+pih, '', '0', '', pih])
                plt.yticks([-90, -45, 0, 45, 90, 135],
                           ['-'+pih, '', '0', '', pih])

                plt.grid()
    #            plt.legend(loc='center left', bbox_to_anchor=(1, .5))
                fig.set_size_inches(10.5, 8)
                plt.axis('image')
                plt.xlim([-90, 90])
                plt.ylim([-135, 160.43])

                name = 'cofig_space_{}_{}_{}.tex'.format(
                        key, startpose, str(x2).replace('.', '_'))
                kwargs = {'extra_axis_parameters':
                      {'anchor=origin', 'x=.02cm', 'y=.02cm'}}
                save_utils.save_plt_as_tikz('Out/analytic_model4/'+name, **kwargs)
#                fig.clear()

# %%

    plt.show()
