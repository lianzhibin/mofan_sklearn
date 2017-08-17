#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:08:00 2017

@author: xiaolian
"""

from sklearn import datasets

import matplotlib.pylab as plt

X, y = datasets.make_regression(n_samples = 100, n_features = 1, n_targets = 1, noise = 10)

plt.scatter(X, y)

plt.show()

