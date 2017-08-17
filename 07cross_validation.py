#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:30:51 2017

@author: xiaolian
"""

from sklearn.datasets import load_iris # iris数据集
from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn.neighbors import KNeighborsClassifier # K最近邻(kNN，k-NearestNeighbor)分类算法

                                                         
                                                         
iris = load_iris()

X = iris.data

y = iris.target


'''
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 4)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))
'''

#cross_val_score
'''
from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv = 5, scoring = 'accuracy')
print(scores)
'''

from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

range_ = range(1, 31)
scores = []

for i in range_:
    knn = KNeighborsClassifier(n_neighbors=i)
    loss = - cross_val_score(knn, X, y, cv = 10, scoring='mean_squared_error')
    acc = cross_val_score(knn, X, y, cv = 10, scoring='accuracy')
    scores.append(acc.mean())
    
plt.plot(range_, scores)
plt.xlabel('Vlaue of K for knn')
plt.ylabel('Cross_validation Accuracy')
plt.show()






                                     


 






















                   