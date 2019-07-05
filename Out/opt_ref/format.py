# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:43:42 2019

@author: AmP
"""

from PIL import Image
import numpy as np
from os import listdir


def crop(png_image_name):
    pil_image = Image.open(png_image_name)
    print('original size: ', pil_image.size)
    np_array = np.array(pil_image)

    blank_px = [255, 255, 255, 255]
    mask = np_array != blank_px
    coords = np.argwhere(mask)
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1
    cropped_box = np_array[x0:x1, y0:y1, z0:z1]
    print(np.size(cropped_box))
    pil_image_ = pil_image.crop((y0, x0, y1, x1))
    print('cropped size: ', pil_image_.size)

    return pil_image_


for f in listdir('.'):
    if f.endswith('.png'):
        f_ = f.split('_')

        method = f_[0]
        n_cyc = f_[1]
        f_len = f_[2]
        f_ori = f_[3]
        f_ang = f_[4]

        print(f_len, f_ori, f_ang)

        img_cropped = crop(f)
        img_cropped.save('cropped/'+f_len+'_'+f_ori+'_'+f_ang+'.png')
