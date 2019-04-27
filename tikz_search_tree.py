# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:25:25 2019

@author: ls
"""


from Src import plot_fun as pf
from Src import kinematic_model as model


# %% POSE 0000
alpha = [0, 0, -0, 0, 0]
eps = 88
F1 = (0, 0)
p0000 = model.set_initial_pose(alpha, eps, F1)
g0000 = pf.GeckoBotGait(p0000)
g0000.plot_gait()
g0000.save_as_tikz('0000')

# %% POSE 1000
ref = [[0, 90, 90, 0, 90], [0, 1, 1, 0]]
g1000 = pf.predict_gait([ref], p0000)
# g1000.plot_gait()
g1000.save_as_tikz('1000')

# %% POSE 0000_
ref = [[0, 0, 0, 0, 0], [1, 0, 0, 1]]
g0000_ = pf.predict_gait([ref], g1000.poses[-1])
g0000_.save_as_tikz('0000_')

# %% POSE 0100
ref = [[90, 0, -90, 90, 0], [1, 0, 0, 1]]
g0100 = pf.predict_gait([ref], p0000)
g0100.plot_gait()
g0100.save_as_tikz('0100')

# %% POSE 0100_
alpha = [0, 90, 90, 0, 90]
eps = 90
F1 = (0, 0)
p1000 = model.set_initial_pose(alpha, eps, F1)

ref = [[90, 0, -90, 90, 0], [1, 0, 0, 1]]
g0100_ = pf.predict_gait([ref], p1000)
g0100_.save_as_tikz('_0100')

# %% POSE 1001
ref = [[40, 1, -10, 60, 10], [1, 0, 0, 1]]
g1001 = pf.predict_gait([ref], p1000)
g1001.save_as_tikz('1001')

# %% POSE 1000_
ref = [[0, 90, 90, 0, 90], [0, 1, 1, 0]]
g1000_ = pf.predict_gait([ref], g1001.poses[-1])
g1000_.save_as_tikz('1000_')

# %% POSE 1002
ref = [[48, 104, 114, 27, 124], [0, 1, 1, 0]]
g1002 = pf.predict_gait([ref], p1000)
g1002.save_as_tikz('1002')

# %% POSE 1003
ref = [[1, 72, 70, 1, 55], [1, 0, 0, 1]]
g1003 = pf.predict_gait([ref], g1002.poses[-1])
g1003.save_as_tikz('1003')

# %% POSE 1002_
ref = [[48, 104, 114, 27, 124], [0, 1, 1, 0]]
g1002_ = pf.predict_gait([ref], g1003.poses[-1])
g1002_.save_as_tikz('1002_')

# %% POSE 1000__
ref = [[0, 90, 90, 0, 90], [0, 1, 1, 0]]
g1000__ = pf.predict_gait([ref], g1003.poses[-1])
g1000__.save_as_tikz('1000__')

# %% POSE 1100
ref = [[45, 45, 0, 45, 45], [1, 0, 0, 1]]
g1100 = pf.predict_gait([ref], p1000)
g1100.save_as_tikz('1100')

# %% POSE 1101
ref = [[45, 45, -90, 45, 45], [1, 1, 0, 0]]
g1101 = pf.predict_gait([ref], g1100.poses[-1])
g1101.save_as_tikz('1101')

# %% POSE 0100__
ref = [[90, 0, -90, 90, 0], [1, 0, 0, 1]]
g0100__ = pf.predict_gait([ref], g1100.poses[-1])
g0100__.save_as_tikz('0100__')

# %% POSE 1100_
ref = [[45, 45, 0, 45, 45], [0, 0, 1, 1]]
g1101 = pf.predict_gait([ref], g1101.poses[-1])
g1101.save_as_tikz('1100_')

# %% POSE 1000___
ref = [[0, 90, 90, 0, 90], [0, 1, 1, 0]]
g1000___ = pf.predict_gait([ref], g1101.poses[-1])
g1000___.save_as_tikz('1000___')

# %% POSE 1004
ref = [[50, 30, 90, 30, 150], [1, 0, 0, 1]]
g1004 = pf.predict_gait([ref], p1000)
g1004.save_as_tikz('1004')

# %% POSE 1005
ref = [[124, 164, 152, 62, 221], [0, 1, 1, 0]]
g1005 = pf.predict_gait([ref], g1004.poses[-1])
g1005.save_as_tikz('1005')

# %% POSE 1006
ref = [[0, 0, 24, 0, 0], [1, 0, 0, 1]]
g1006 = pf.predict_gait([ref], g1005.poses[-1])
g1006.save_as_tikz('1006')

# %% POSE 1005
ref = [[124, 164, 152, 62, 221], [0, 1, 1, 0]]
g1005_ = pf.predict_gait([ref], g1006.poses[-1])
g1005_.save_as_tikz('1005_')

# %% POSE 1007
ref = [[30, 90, 80, 10, 10], [1, 0, 0, 1]]
g1007 = pf.predict_gait([ref], g1005.poses[-1])
g1007.save_as_tikz('1007')

# %% POSE 1000____
ref = [[1, 90, 90, 1, 90], [0, 1, 1, 0]]
g1000____ = pf.predict_gait([ref], g1007.poses[-1])
g1000____.save_as_tikz('1000____')















# %% LEFT CURVES ############################################

# %% POSE 0000_
ref = [[0, 0, 0, 0, 0], [0, 1, 1, 0]]
g0000__ = pf.predict_gait([ref], g0100.poses[-1])
g0000__.save_as_tikz('0000__')

# %% POSE _1000
alpha = [90, 0, -90, 90, 0]
eps = 90
F1 = (0, 0)
p0100 = model.set_initial_pose(alpha, eps, F1)

ref = [[0, 90, 90, 0, 90], [0, 1, 1, 0]]
g1000_ = pf.predict_gait([ref], p0100)
g1000_.save_as_tikz('_1000')

################
# %% POSE 0101
ref = [[1, 40, 10, 10, 60], [0, 1, 1, 0]]
g0101 = pf.predict_gait([ref], p0100)
g0101.save_as_tikz('0101')

# %% POSE 0100_
ref = [[90, 0, -90, 90, 0], [1, 0, 0, 1]]
g0100_ = pf.predict_gait([ref], g0101.poses[-1])
g0100_.save_as_tikz('0100_')

################
# %% POSE 0102
ref = [[104, 48, -114, 124, 27], [1, 0, 0, 1]]
g0102 = pf.predict_gait([ref], p0100)
g0102.save_as_tikz('0102')

# %% POSE 0103
ref = [[72, 1, -70, 55, 1], [0, 1, 1, 0]]
g0103 = pf.predict_gait([ref], g0102.poses[-1])
g0103.save_as_tikz('0103')

# %% POSE 0102_
ref = [[104, 48, -114, 124, 27], [1, 0, 0, 1]]
g0102_ = pf.predict_gait([ref], g0103.poses[-1])
g0102_.save_as_tikz('0102_')

# %% POSE 0100__
ref = [[90, 0, -90, 90, 0], [1, 0, 0, 1]]
g0100__ = pf.predict_gait([ref], g0103.poses[-1])
g0100__.save_as_tikz('0100__')

################
# %% POSE 1100_
ref = [[45, 45, 0, 45, 45], [0, 1, 1, 0]]
g1100__ = pf.predict_gait([ref], p1000)
g1100__.save_as_tikz('1100__')

# %% POSE 1101
ref = [[45, 45, 90, 45, 45], [1, 1, 0, 0]]
g1102 = pf.predict_gait([ref], g1100__.poses[-1])
g1102.save_as_tikz('1102')

# %% POSE 1100___
ref = [[45, 45, 0, 45, 45], [0, 0, 1, 1]]
g1100___ = pf.predict_gait([ref], g1102.poses[-1])
g1100___.save_as_tikz('1100___')

# %% POSE 0100___
ref = [[0, 90, 90, 0, 90], [0, 1, 1, 0]]
g0100___ = pf.predict_gait([ref], g1100___.poses[-1])
g0100___.save_as_tikz('0100___')


################
# %% POSE 0104
ref = [[30, 50, -90, 150, 30], [0, 1, 1, 0]]
g0104 = pf.predict_gait([ref], p1000)
g0104.save_as_tikz('0104')

# %% POSE 0105
ref = [[164, 124, -152, 221, 62], [1, 0, 0, 1]]
g0105 = pf.predict_gait([ref], g0104.poses[-1])
g0105.save_as_tikz('0105')

# %% POSE 0106
ref = [[0, 0, -24, 0, 0], [0, 1, 1, 0]]
g0106 = pf.predict_gait([ref], g0105.poses[-1])
g0106.save_as_tikz('0106')

# %% POSE 0105
ref = [[164, 124, -152, 221, 62], [1, 0, 0, 1]]
g0105_ = pf.predict_gait([ref], g0106.poses[-1])
g0105_.save_as_tikz('0105_')

# %% POSE 0107
ref = [[90, 30, -80, 10, 10], [0, 1, 1, 0]]
g0107 = pf.predict_gait([ref], g0105.poses[-1])
g0107.save_as_tikz('0107')

# %% POSE 0100____
ref = [[90, 0, -90, 90, 0], [1, 0, 0, 1]]
g0100____ = pf.predict_gait([ref], g0107.poses[-1])
g0100____.save_as_tikz('0100____')




