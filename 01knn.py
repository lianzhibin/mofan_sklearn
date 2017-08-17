#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 21:02:32 2017

@author: xiaolian
"""

'''
steps:
    
    1、load model
    2、creat data
    3、build model - train - predict
    


'''

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_Y, test_size = 0.3)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(y_test)