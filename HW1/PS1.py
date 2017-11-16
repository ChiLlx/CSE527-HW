#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:13:35 2017

@author: yingxc
"""

from __future__ import division, absolute_import, print_function
import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import K2Score
#%%

#Method 1

# Read data
data = pd.read_table('disc-gal80-gal4-gal2.txt', index_col=0).values

# Get params for Model 1
M = len(data[0])
Ma1 = np.count_nonzero(data[0])
Ma0 = M - Ma1
Mb1 = np.count_nonzero(data[1])
Mb0 = M - Mb1
Mb_a1 = np.count_nonzero(data[1][np.where(data[0]==1)])
Mb_a0 = np.count_nonzero(data[1][np.where(data[0]==0)])
Mc_b1 = np.count_nonzero(data[2][np.where(data[1]==1)])
Mc_b0 = np.count_nonzero(data[2][np.where(data[1]==0)])

theta_a = Ma1 / M
theta_b_1 = Mb_a1 / Ma1
theta_b_0 = Mb_a0 / Ma0
theta_c_1 = Mc_b1 / Mb1
theta_c_0 = Mc_b0 / Mb0

#Compute log Likelihood
def local_l(theta, M, M_):

    return M_ * np.log(theta) + (M - M_) * np.log(1 - theta)

L1 = np.sum(local_l(np.array([theta_a, theta_b_1, theta_b_0, theta_c_1, theta_c_0]),
                        np.array([M, Ma1, Ma0, Mb1, Mb0]),
                        np.array([Ma1, Mb_a1, Mb_a0, Mc_b1, Mc_b0])))
print ('log likelilhood for model 1 is ' + str(L1))
#%%
# Get params for Model 2
M = len(data[0])
Ma1 = np.count_nonzero(data[0])
#Ma0 = M - Ma1
Mb1 = np.count_nonzero(data[1])
#Mb0 = M - Mb1
M11 = len(np.intersect1d(np.where(data[0]==1)[0], np.where(data[1]==1)[0]))
M10 = len(np.intersect1d(np.where(data[0]==1)[0], np.where(data[1]==0)[0]))
M01 = len(np.intersect1d(np.where(data[0]==0)[0], np.where(data[1]==1)[0]))
M00 = len(np.intersect1d(np.where(data[0]==0)[0], np.where(data[1]==0)[0]))
Mc_11 = np.count_nonzero(data[2][np.intersect1d(np.where(data[0]==1)[0], np.where(data[1]==1)[0])])
Mc_10 = np.count_nonzero(data[2][np.intersect1d(np.where(data[0]==1)[0], np.where(data[1]==0)[0])])
Mc_01 = np.count_nonzero(data[2][np.intersect1d(np.where(data[0]==0)[0], np.where(data[1]==1)[0])])
Mc_00 = np.count_nonzero(data[2][np.intersect1d(np.where(data[0]==0)[0], np.where(data[1]==0)[0])])

theta_a = Ma1 / M
theta_b = Mb1 / M
theta_c_11 = Mc_11 / M11
theta_c_10 = Mc_10 / M10
theta_c_01 = Mc_01 / M01
theta_c_00 = Mc_00 / M00

#Compute Likelihood
L2 = np.sum(local_l(np.array([theta_a, theta_b, theta_c_11, theta_c_10, theta_c_01, theta_c_00]),
                       np.array([M, M, M11, M10, M01, M00]),
                       np.array([Ma1, Mb1, Mc_11, Mc_10, Mc_01, Mc_00])))
print ('log likelilhood for model 2 is ' + str(L2))
#%%
#Method 2, Using pgmpy

# Read Data
train_data = pd.DataFrame(pd.read_table('disc-gal80-gal4-gal2.txt', index_col=0).values.T, columns=['Gal80', 'Gal4', 'Gal2'])

# Define Model
model1 = BayesianModel([('Gal80', 'Gal4'), ('Gal4', 'Gal2')])
model2 = BayesianModel([('Gal80', 'Gal2'), ('Gal4', 'Gal2')])

# Fit the data
model1.fit(train_data)
model2.fit(train_data)

# Get CPDs
print('For Model 1')
print(model1.get_cpds('Gal80'))
print(model1.get_cpds('Gal4'))
print(model1.get_cpds('Gal2'))

print('For Model 2')
print(model2.get_cpds('Gal80'))
print(model2.get_cpds('Gal4'))
print(model2.get_cpds('Gal2'))

#Calculate K2 Score
print('K2Score of Model1 and 2')
print(K2Score(train_data).score(model1))
print(K2Score(train_data).score(model2))
