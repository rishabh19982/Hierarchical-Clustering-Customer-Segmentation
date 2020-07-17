# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:47:03 2020

@author: Administrator
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Customers.csv')
x = dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))

plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3,
                             affinity='euclidean',
                             linkage='ward')

y_hc = hc.fit_predict(x)

# Visualising the clusters and interpretation
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'cyan', label = '1st Cluster')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'green', label = '2nd Cluster')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'red', label = '3rd Cluster')
#plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'blue', label = '4th Cluster')

plt.title('Clusters of customers')
plt.xlabel('Annual Salary (k$)')
plt.ylabel('Spendings (1 to 100)')
plt.legend()
plt.show()