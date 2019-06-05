# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:59:53 2019

@author: Hulk
"""


import csv
import numpy as np
import os


def adjust_filename(filename):
    dirname = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))))
    realpath = filename
    filename = os.path.join(dirname, realpath)
    filename = os.path.abspath(os.path.realpath(filename))
    return filename


def save_sample_as_csv(data, filename):
    filename = adjust_filename(filename)
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
    return filename


def read_csv(filename):
    filename = adjust_filename(filename)
    dic = {}
    mapping = {}
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for idx, row in enumerate(reader):
            # print ', '.join(row)
            if idx == 0:
                for jdx, key in enumerate(row):
                    mapping[jdx] = key
                    dic[key] = []
            else:
                for jdx, val in enumerate(row):
                    try:
                        dic[mapping[jdx]].append(float(val) if val else np.nan)
                    except ValueError:
                        if '[' in val:
                            val = val.replace('[ ', '')
                            val = val.replace('[', '')
                            val = val.replace('  ]', '')
                            val = val.replace(' ]', '')
                            val = val.replace(']', '')
                            val = val.replace(',', '')
                            val = val.replace('   ', '  ')
                            val = val.replace('  ', ' ')
                            lis = val.split(' ')
                            val = [float(l) for l in lis]
                            dic[mapping[jdx]].append(val)
                        else:
                            dic[mapping[jdx]].append(val)
    return dic


if __name__ == '__main__':
    filename = save_sample_as_csv({'x0': [1], 'x2': [1]}, 'Out/test2.csv')
    dic = read_csv('Out/test2.csv')
    print dic
