#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:30:28 2017

@author: yingxc
"""

from __future__ import division
import pandas as pd
import numpy as np
#from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
#%%
###########Problem 1############

#Read data
genotype = pd.read_table('https://sites.google.com/a/cs.washington.edu/cse527-au17/genotype.txt?attredirects=0&d=1', header=None)
phenotype = pd.read_table('https://sites.google.com/a/cs.washington.edu/cse527-au17/phenotype.txt?attredirects=0&d=1', header=None)

phenotype = phenotype.values
genotype = genotype.values
#%%
#Calculate LOD scores
def norm(mu, sigma, x):

    return 1 / np.sqrt(2*np.pi*sigma**2) * np.exp(-np.power(x - mu, 2) / (2 * sigma**2))

def LOD(gentype, QTL):
    ''' Calculate the LOD score for a given gene
    '''
    gen1 = QTL[gentype.astype(np.bool)]
    gen0 = QTL[np.logical_not(gentype.astype(np.bool))]
    mu1 = np.mean(gen1)
    mu0 = np.mean(gen0)
    mu = np.mean(QTL)
#    sigma_10 = (np.var(gen1) * len(gen1) + np.var(gen0) * len(gen0)) / len(QTL)   #Assuming equal var for differnet genotypes
    sigma_1 = np.std(gen1)
    sigma_0 = np.std(gen0)
    sigma = np.std(QTL)
#    dist1 = norm(mu1, sigma_1)
#    dist0 = norm(mu0, sigma_0)
#    dist = norm(mu, sigma)
#    logQTL = np.sum(np.log10(dist1.pdf(gen1))) + np.sum(np.log10(dist0.pdf(gen0)))
#    lognoQTL = np.sum(np.log10(dist.pdf(gentype)))
    logQTL = np.sum(np.log10(norm(mu1, sigma_1, gen1))) + np.sum(np.log10(norm(mu0, sigma_0, gen0)))
    lognoQTL = np.sum(np.log10(norm(mu, sigma, QTL)))
    score = logQTL - lognoQTL

    return score

def max_LOD(gentype, QTL):
    LODscores = []
    for i in range(len(genotype)):
        LODscores.append(LOD(genotype[i], phenotype[0]))

    return max(LODscores)

LODscores = []
for i in range(len(genotype)):
    LODscores.append(LOD(genotype[i], phenotype[0]))
print('Max LOD is ' + str(max_LOD(genotype, phenotype)))
#%%
permutation = 500
max_list = []
for i in range(permutation):
    np.random.shuffle(phenotype[0])
    max_list.append(max_LOD(genotype, phenotype))

#%%
threshold = np.sort(max_list)[int(0.95 * len(max_list))]
print('The threshold is ' + str(threshold))

fig = plt.figure(figsize=(8, 6))
sns.distplot(max_list)
plt.title('Distribution of Max LOD Scores for Permutation Test')
plt.xlabel('Max LOD Scores')
plt.ylabel('Frequency')
fig.savefig('../Q1_b', dpi=400)

print ('genes that over the threshold are ' + str(np.where(LODscores > threshold)[0]))

#%%
###########Problem 2############

#Normalize data
geno_norm = scale(genotype.T)

#Split the training and test set
cut = 250
train_geno = geno_norm[0:cut, :]
test_geno = geno_norm[cut:, :]
train_pheno = phenotype.T[0:cut, :]
test_pheno = phenotype.T[cut:, :]

def reg_model(train_X, test_X, train_y, test_y, alpha_range, r = "lasso"):

    mse = []
    for i in alpha_range:
        if(r == "lasso"):
            clf = linear_model.Lasso(alpha = i)
        elif(r == "ridge"):
            clf = linear_model.Ridge(alpha = i)
        clf.fit(train_X, train_y)
        pred = clf.predict(test_X)
        mse.append(mean_squared_error(pred, test_y))

    return mse

alpht_range = np.logspace(-2, 4, 1000)
l1_mse = reg_model(train_geno, test_geno, train_pheno, test_pheno,
                       alpht_range, r='lasso')

l2_mse = reg_model(train_geno, test_geno, train_pheno, test_pheno,
                       alpht_range, r="ridge")

fig = plt.figure(figsize=(8,6))
plt.plot(alpht_range, l1_mse, label='L1')
plt.plot(alpht_range, l2_mse, label='L2')
plt.xscale('log')
plt.legend()
plt.xlabel(r'$\alpha$')
plt.ylabel('MSE')
fig.savefig('../Q2_a', dpi=400)

print ('min MSE for lasso is ' + str(min(l1_mse)))
print ('min MSE for ridge is ' + str(min(l2_mse)))

#%%
#LOOCV
alpht_range1 = np.logspace(-1.5, -0.5, 12)
alpht_range2 = np.logspace(3.5, 4.5, 12)

mse_list1 = np.empty((len(phenotype[0]), len(alpht_range1)))
mse_list2 = np.empty((len(phenotype[0]), len(alpht_range2)))
for i in range(len(phenotype[0])):
    train_geno = np.delete(geno_norm, i, axis=0)
    test_geno = geno_norm[i].reshape(1, 1333)
    train_pheno = np.delete(phenotype, i).T.reshape(333, 1)
    test_pheno = phenotype[0, i].reshape(1, 1)
    l1_mse = reg_model(train_geno, test_geno, train_pheno, test_pheno,
                       alpht_range1, r='lasso')

    l2_mse = reg_model(train_geno, test_geno, train_pheno, test_pheno,
                       alpht_range2, r="ridge")
    mse_list1[i] = l1_mse
    mse_list2[i] = l2_mse

l1_mse = np.average(mse_list1, axis=0)
l2_mse = np.average(mse_list2, axis=0)
fig = plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(alpht_range1, l1_mse, label='L1')
#plt.xscale('log')
plt.legend()
plt.xlabel(r'$\alpha$')
plt.ylabel('MSE')
plt.title('LOOCV')
plt.subplot(122)
plt.plot(alpht_range2, l2_mse, label='L2')
#plt.xscale('log')
plt.legend()
plt.xlabel(r'$\alpha$')
plt.ylabel('MSE')
plt.title('LOOCV')
fig.savefig('../Q2_c', dpi=400)

print ('min MSE for lasso is ' + str(min(l1_mse)))
print ('min MSE for ridge is ' + str(min(l2_mse)))
#%%
###########Problem 3############

#Define Zij to store prop
class Zij:
    p = 0;
    x_idx = 0;
    y_idx = 0;
    def __init__(self, p, x_idx, y_idx):
        self.p = float(p)
        self.x_idx = int(x_idx)
        self.y_idx = int(y_idx)

#initialize Z values and freq
Z = np.array([[Zij(0.5, 2, 6), Zij(0.5, 4, 5)], [Zij(0.5, 0, 4), Zij(0.5, 1, 3)], [Zij(0.5, 4, 8), Zij(0.5, 6, 7)]])
freq_1 = np.array([1/12, 1/12, 1/12, 1/12, 3/12, 1/12, 2/12, 1/12, 1/12])
freq_2 = np.zeros(9)
error_list = []
error = float(1000)

max_time = 1000 # max run time
d = 1e-5        # for convergence

while(error > d and len(error_list) < max_time):
    
    #E step
    print('------------------------------------------------------'
          + '\nIteration: ' + str(len(error_list)) + '\nProbability:')
    for htype in Z:
        tmp_0 = freq_1[htype[0].x_idx] * freq_1[htype[0].y_idx]
        tmp_1 = freq_1[htype[1].x_idx] * freq_1[htype[1].y_idx]
        htype[0].p = tmp_0 / (tmp_0 + tmp_1)
        htype[1].p = tmp_1 / (tmp_0 + tmp_1)
        print(str(htype[0].p) + ',' + str(htype[1].p))
        
    #M step
    freq_2 = np.zeros(9)
    for htype in Z:
        for i in range(htype.size):
            freq_2[htype[i].x_idx] += htype[i].p / Z.size
            freq_2[htype[i].y_idx] += htype[i].p / Z.size
    print("Frequency:")
    print(freq_2)
    error = np.sum(abs(freq_1 - freq_2))
    error_list.append(error)
    freq_1 = freq_2
    
print('Finished!')
print('--------------------------------------------------------')
print("Frequency:")
print(np.round(freq_2, 3))
print('Probability:')
for htype in Z:
    print(str(np.round(htype[0].p, 2)) + ',' + str(np.round(htype[1].p, 2)))
