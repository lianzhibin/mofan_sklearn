#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:06:29 2017

@author: xiaolian
"""

# 标准化 数据模块

from sklearn import preprocessing
import numpy as np

# 
from sklearn.model_selection import train_test_split

# 
from sklearn.svm import SVC

# 
import matplotlib.pyplot as plt


from sklearn.datasets.samples_generator import make_classification

X, y = make_classification(n_samples = 300, n_features = 2, n_redundant = 0, n_informative = 2, random_state = 22, \
                           n_clusters_per_class = 1, scale = 100       \
                           )

X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

plt.scatter(X[:, 0], X[:, 1], c = y)

plt.show()