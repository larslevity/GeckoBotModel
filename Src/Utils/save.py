# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:18:20 2019

@author: AmP
"""

from matplotlib2tikz import save as tikz_save
import fileinput
from PIL import Image, ImageChops
import os

import sys





def save_plt_as_tikz(filename, gecko_str=None, scale=1, **kwargs):
    print('Saving as TikZ-Picture...')
    aux_fn = filename + '_aux'
    if gecko_str:
        kwargs = {'extra_axis_parameters':
                  {'anchor=origin', 'disabledatascaling', 'x=1cm', 'y=1cm'}}
    tikz_save(aux_fn, **kwargs)
    insert_tex_header(aux_fn, gecko_str, scale)

    # remove blank lines:
    with open(aux_fn, 'r') as file:
        if sys.version_info[0] < 3:
            with open(filename, 'wb') as ofile:
                for line in file:
                    if not line == '\n':
                        ofile.write(line)
        else:
            try:
                with open(filename, 'x') as ofile:
                    for line in file:
                        if not line == '\n':
                            ofile.write(line)
            except FileExistsError:
                with open(filename, 'w') as ofile:
                    for line in file:
                        if not line == '\n':
                            ofile.write(line)
    os.remove(aux_fn)
    print('Done!')


def save_geckostr_as_tikz(filename, geckostr):
    header = """
\\documentclass[crop,tikz]{standalone}
\\usepackage[utf8]{inputenc}
\\usepackage{tikz}
\\begin{document}
\\begin{tikzpicture}[scale=1]
"""
    ending = """
%% End geckostr %%
\\end{tikzpicture}
\\end{document}
"""
    with open(filename, 'w') as fout:
            fout.writelines(header + geckostr + ending)


def insert_tex_header(filename, gecko_str=None, scale=1):
    header = """
\\documentclass[crop,tikz]{standalone}
\\usepackage[utf8]{inputenc}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\pgfplotsset{compat=newest}
\\usepgfplotslibrary{groupplots}
\\begin{document}
"""
    if gecko_str:
        # remove \begin{tikzpicture}
        with open(filename, 'r') as fin:
            data = fin.read().splitlines(True)
        with open(filename, 'w') as fout:
            fout.writelines([data[0]] + data[2:])
        # add geckostr between header and matplotlib2tikz data
        header = header + '\n\\begin{tikzpicture}[scale=%s]' % (scale) + gecko_str
    line_pre_adder(filename, header)
    # Append Ending
    ending = "\n%% End matplotlib2tikz content %% \n\\end{document}"
    with open(filename, "a") as myfile:
        myfile.write(ending)


def line_pre_adder(filename, line_to_prepend):
    f = fileinput.input(filename, inplace=1)
    for xline in f:
        if f.isfirstline():
            print(line_to_prepend.rstrip('\r\n') + '\n' + xline,)
        else:
            print(xline,)


def trim(filename, border=1):
    im = Image.open(filename)
    bg = Image.new(im.mode, im.size, border)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        im2 = im.crop(bbox)
        im2.save(filename)

def crop_img(filename):
    im = Image.open(filename)
    print(im.getbbox())
    im2 = im.crop(im.getbbox())
    im2.save(filename)
