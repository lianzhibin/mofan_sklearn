#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 19:05:45 2017

@author: xiaolian
"""

from sklearn import svm
from sklearn import datasets



'''

df = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

'''


'''
# method 1 : pickle
import pickle
'''

# 
'''
with open('clf.pickle', 'wb') as f:
    pickle.dump(clf, f)
'''

'''
iris = datasets.load_iris()
X, y = iris.data, iris.target

with open('clf.pickle', 'rb') as f:
    clf2 = pickle.load(f)
    print(clf2.predict(X[0:2]))
'''  
    
   
# method2 :joblib

from sklearn.externals import joblib

# Save
#joblib.dump(clf, 'clf.pkl')

# restore
clf3 = joblib.load('clf.pkl')
print(clf3.predict(X[0:3]))



























 