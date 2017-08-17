#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 12:43:11 2017

@author: xiaolian
"""

from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()

model.fit(data_X, data_y)

print(model.predict(data_X[:4, :]))

print(model.coef_) # 斜率
print(model.intercept_) # intercept
print(model.get_params())

print(model.score(data_X, data_y))



     

