#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 21:54:59 2017

@author: xiaolian
"""

from __future__ import print_function

from sklearn import datasets

from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()

data_X = loaded_data.data

data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

print(model.predict(data_X[:4, :]))
print(data_y[:4])

