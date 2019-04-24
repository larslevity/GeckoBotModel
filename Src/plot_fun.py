# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:38:56 2019

@author: ls
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import kinematic_model as model



def plot_gait(data_xy, data_fp, data_nfp, data_x):
    for idx in range(len(data_xy)):
        (x, y) = data_xy[idx]
        (fpx, fpy) = data_fp[idx]
        (nfpx, nfpy) = data_nfp[idx]
        c = (1-float(idx)/len(data_xy))*.8
        col = (c, c, c)
        plt.plot(x, y, '.', color=col)
        plt.plot(fpx, fpy, 'o', markersize=10, color=col)
        plt.plot(nfpx, nfpy, 'x', markersize=10, color=col)
    plt.axis('equal')


def start_end(data_xy, data_fp, data_nfp, data_x):
    return ([data_xy[0], data_xy[-1]], [data_fp[0], data_fp[-1]],
            [data_nfp[0], data_nfp[-1]], [data_x[0], data_x[-1]])


def start_mid_end(data_xy, data_fp, data_nfp, data_x):
    mid = len(data_xy)/2
    return ([data_xy[0], data_xy[mid], data_xy[-1]],
            [data_fp[0], data_fp[mid], data_fp[-1]],
            [data_nfp[0], data_nfp[mid], data_nfp[-1]],
            [data_x[0], data_x[mid], data_x[-1]])


def animate_gait(fig1, data_xy, data_markers, inv=500,
                 col=['red', 'orange', 'green', 'blue', 'magenta', 'darkred']):

    def update_line(num, data_xy, line_xy, data_markers,
                    lm0, lm1, lm2, lm3, lm4, lm5, leps):
        x, y = data_xy[num]
        line_xy.set_data(np.array([[x], [y]]))

        xm0, ym0 = data_markers[num][0][0], data_markers[num][1][0]
        xm1, ym1 = data_markers[num][0][1], data_markers[num][1][1]
        xm2, ym2 = data_markers[num][0][2], data_markers[num][1][2]
        xm3, ym3 = data_markers[num][0][3], data_markers[num][1][3]
        xm4, ym4 = data_markers[num][0][4], data_markers[num][1][4]
        xm5, ym5 = data_markers[num][0][5], data_markers[num][1][5]
        lm0.set_data(np.array([[xm0], [ym0]]))
        lm1.set_data(np.array([[xm1], [ym1]]))
        lm2.set_data(np.array([[xm2], [ym2]]))
        lm3.set_data(np.array([[xm3], [ym3]]))
        lm4.set_data(np.array([[xm4], [ym4]]))
        lm5.set_data(np.array([[xm5], [ym5]]))
        leps.set_data(np.array([[xm1, xm4], [ym1, ym4]]))
        return line_xy, lm0, lm1, lm2, lm3, lm4, lm5, leps

    n = len(data_xy)
    l_xy, = plt.plot([], [], 'k.', markersize=3)
    msize = 5
    lm0, = plt.plot([], [], 'o', color=col[0], markersize=msize)
    lm1, = plt.plot([], [], 'o', color=col[1], markersize=msize)
    lm2, = plt.plot([], [], 'o', color=col[2], markersize=msize)
    lm3, = plt.plot([], [], 'o', color=col[3], markersize=msize)
    lm4, = plt.plot([], [], 'o', color=col[4], markersize=msize)
    lm5, = plt.plot([], [], 'o', color=col[5], markersize=msize)
    leps, = plt.plot([], [], '-', color='mediumpurple', linewidth=1)

    minx, maxx, miny, maxy = 0, 0, 0, 0
    for dataset in data_markers:
        x, y = dataset
        minx = min(x) if minx > min(x) else minx
        maxx = max(x) if maxx < max(x) else maxx
        miny = min(y) if miny > min(y) else miny
        maxy = max(y) if maxy < max(y) else maxy
    plt.xlim(minx-5, maxx+5)
    plt.ylim(miny-5, maxy+5)
    line_ani = animation.FuncAnimation(
        fig1, update_line, n, fargs=(data_xy, l_xy, data_markers,
                                     lm0, lm1, lm2, lm3, lm4, lm5, leps),
        interval=inv, blit=True)
    return line_ani


def save_animation(line_ani, name='gait.mp4', conv='avconv'):
    """
    To create gif:
        0. EASY: Use : https://ezgif.com/video-to-gif
        OR:
        1. Create a directory called frames in the same directory with
           your .mp4 file. Use command:
            ffmpeg -i video.mp4  -r 5 'frames/frame-%03d.jpg'

            -r 5 stands for FPS value
                for better quality choose bigger number
                adjust the value with the -delay in 2nd step
                to keep the same animation speed

            %03d gives sequential filename number in decimal form

        1a. Loop the thing (python):
            import os
            for jdx, idx in enumerate(range(1, 114)[::-1]):
                os.rename('frame-'+'{}'.format(idx).zfill(3)+'.jpg',
                          'frame-'+'{}'.format(114+jdx).zfill(3)+'.jpg')

        1b. Reduce size of single frames (bash):
            for i in *.jpg; do convert "$i" -quality 20 "${i%%.jpg*}_new.jpg"; done

        2. Convert Images to gif (bash):
            cd frames
            convert -delay 20 -loop 0 *.jpg myimage.gif

            -delay 20 means the time between each frame is 0.2 seconds
               which match 5 fps above.
               When choosing this value
                   1 = 100 fps
                   2 = 50 fps
                   4 = 25 fps
                   5 = 20 fps
                   10 = 10 fps
                   20 = 5 fps
                   25 = 4 fps
                   50 = 2 fps
                   100 = 1 fps
                   in general 100/delay = fps

            -loop 0 means repeat forever
        2a. To further compress you can skip frames:
            gifsicle -U input.gif `seq -f "#%g" 0 2 99` -O2 -o output.gif

    """
    # Set up formatting for the movie files
    Writer = animation.writers[conv]
    writer = Writer(fps=15, metadata=dict(artist='Lars Schiller'),
                    bitrate=1800)
    line_ani.save(name, writer=writer)


if __name__ == "__main__":
    """
    To save the animation you need the libav-tool to be installed:
    sudo apt-get install libav-tools
    """

    col = ['red', 'orange', 'green', 'blue', 'magenta', 'darkred']


#   init_pose = [(alp)          , eps, pos foot1]
    init_pose = [(90, 1, -90, 90, 1), 0, (0, 0)]

    ref = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [0, 1, 1, 0]]
           for gam in range(-90, 91, 45)]
    ref2 = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [1, 0, 0, 1]]
            for gam in range(-90, 90, 45)[::-1]]  # revers
    ref = ref + ref2

    x, r, data, cst, marks = model.predict_pose(ref, init_pose, True, False)

#    plt.figure()
#    plot_gait(*start_end(*data))
#
#    markers = marker_history(marks)
#    for idx, marker in enumerate(markers):
#        x, y = marker
#        plt.plot(x, y, color=col[idx])

    # ## withot stretching
    init_pose = [(90, 1, -90, 90, 1), 0, (0, 0)]
    step = 45
    ref = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [0, 1, 0, 0]]
           for gam in range(-90, 91, step)]
    ref2 = [[[45-gam/2., 45+gam/2., gam, 45-gam/2., 45+gam/2.], [1, 0, 0, 0]]
            for gam in range(-90, 90, step)[::-1]]  # revers
    ref = ref + ref2

    x, r, data, cst, marks = model.predict_pose(ref, init_pose, True, False,
                                          dev_ang=.1)

    plt.figure()
    plot_gait(*start_mid_end(*data))

    markers = model.marker_history(marks)
    for idx, marker in enumerate(markers):
        x, y = marker
        plt.plot(x, y, '-', color=col[idx])

    # Animation
    data_xy = data[0]
    fig_ani = plt.figure()
    plt.title('Test')
    _ = animate_gait(fig_ani, data_xy, marks)  # _ = --> important

    for idx, marker in enumerate(markers):
        x, y = marker
        plt.plot(x, y, '-', color=col[idx])

    plt.show()

