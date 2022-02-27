from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 18:30:27 2021

@author: Shahzeb
"""

import numpy as np
import random as rand
import matplotlib.pyplot as plt
import csv



# set a random seed
rand.seed(33)
# ------------------------------------------------
# Generate our true value for this test
# ------------------------------------------------
samples = 100  # number of points in each cluster

# means
mean1 = np.array([3, 70], dtype= np.int64)
mean2 = np.array([7, 150],dtype= np.int64)
mean3 = np.array([13, 250],dtype= np.int64)
"""
# Finding the data
D1 = np.random.normal(loc=mean1, scale=3, size=(100,2))
print(D1.shape)
D2 = np.random.normal(loc=mean2, scale=3, size=(100,2))
D3 = np.random.normal(loc=mean3, scale=3, size=(100,2))
Data = np.concatenate((D1, D2), axis = 0)
Data = np.concatenate((Data, D3), axis = 0)
print (Data)
print(Data.shape)

noise_mean = np.array([0, 0])
noise_dev = np.array([[1, 0], [0, 1]])
D4 = np.random.normal(loc=0, scale=1, size=(300,2))
Data = Data+D4
print (Data)
print(Data.shape)

with open('data.csv','w', newline='') as f:
    write = csv.writer(f)
    write.writerows(Data) """
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('data.csv',  names=['V1', 'V2'])
print("Input Data and Shape")
print(data.shape)
data.head()


fig, ax = plt.subplots()


# Getting the values and plotting it
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)

# Number of clusters
k=3
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(f1), size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(f2), size=k)

C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print("Initial Centroids")
print(C)
# Plotting along with the Centroids
plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='+', s=200, c='m')


# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# Loop will run till the error becomes zero
t=0
while error != 0:
    t=t+1
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distance = dist(X[i], C)
        cluster = np.argmin(distance)
        clusters[i] = cluster
    # Storing the old centroid values
    
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)
    print(error)
    plt.scatter(C[:, 0], C[:, 1], marker='+', s=200, c='c')
    if t > 100:
        break
print(t)

colors = ['r', 'g', 'b', 'y', 'c', 'm']


for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='m')
