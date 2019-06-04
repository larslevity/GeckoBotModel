# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:59:53 2019

@author: Hulk
"""


import csv
import numpy as np
import os
from time import strftime


def save_sample_as_csv(data, filename):

    dirname = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))))
    print(dirname)
#    dirname = os.path.realpath(__file__)
    realpath = filename
    filename = os.path.join(dirname, realpath)
    filename = os.path.abspath(os.path.realpath(filename))

    d = {key: data[key][-1] for key in data}
    keys = sorted(d.keys())

    if not os.path.isfile(filename):
        print("New file created")
        with open(filename, "w") as outfile:
            writer = csv.writer(outfile, delimiter="\t")
            writer.writerow(keys)

    with open(filename, "a") as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        writer.writerow([d[key] for key in keys])
        
        
if __name__ == '__main__':
    save_sample_as_csv({'x0': [1], 'x2': [1]}, 'Out/test2.csv')