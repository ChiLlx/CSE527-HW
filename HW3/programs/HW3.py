#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 20:48:59 2017

@author: yingxc
"""

from __future__ import division
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io

#%%

#import data
data = pd.read_table('../data/mnist_data.txt', header=None, sep=' ').values.reshape(55000,28,28)
labels = pd.read_table('../data/mnist_labels.txt', header=None, sep=' ').values

#create training and validation set
train_data = data[:50000]
valid_data = data[50000:]
train_labels = labels[:50000]
valid_labels = labels[50000:]
#%%

#a) show a random fig
r = np.random.choice(range(len(train_data)))
fig = plt.figure(figsize=(6, 6))
io.imshow(train_data[r])
fig.savefig('../pics/Q4_a')
#%%

model = Sequential()

# First layer 
model.add(Dense(28, input_dim=784))
model.add(Activation('tanh'))

# First layer 
model.add(Dense(15))
model.add(Activation('relu'))

# Second layer
model.add(Dense(10))
model.add(Activation('softmax'))

# Define loss
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#%%

history = model.fit(train_data.reshape(50000, 28**2), train_labels, epochs=10, verbose=1)
#%%

fig = plt.figure(figsize = (8, 6))
plt.plot(history.history["loss"])
plt.xlim(0, 10)
plt.xlabel('Epoch')
plt.ylabel('Loss')
fig.show()
fig.savefig('../pics/Q4_b', dpi=400)
#%%


