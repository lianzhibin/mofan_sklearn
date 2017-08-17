#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 18:56:22 2017

@author: xiaolian
"""


from sklearn.learning_curve import validation_curve #学习曲线模块
from sklearn.datasets import load_digits #digits数据集
from sklearn.svm import SVC #Support Vector Classifier
import matplotlib.pyplot as plt #可视化模块
import numpy as np


param_range = np.logspace(-6, -2.3, 5)


digits = load_digits()
X = digits.data
y = digits.target

train_loss, test_loss = validation_curve( SVC(), X, y, param_name = 'gamma', param_range = param_range, cv =10, scoring = 'neg_mean_squared_error', 
                                                     )


# 平均每一轮所得到的平均方差
train_loss_mean = -np.mean(train_loss, axis = 1)
test_loss_mean = -np.mean(test_loss, axis = 1)

plt.plot(param_range, train_loss_mean, 'o-', color = 'r', label = 'Training')
plt.plot(param_range, test_loss_mean, 'o-', color = 'g', label = 'Corss-validation')

plt.xlabel('Training examples')
plt.ylabel('Loss')
plt.show()


