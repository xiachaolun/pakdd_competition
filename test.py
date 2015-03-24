import os, sys
from pprint import pprint
import math

from datetime import datetime
import time

import numpy as np

import random

import copy

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

X = np.array([[0.1, 0.5, 0],
              [0.3, -0.4, 4],
              [-0.3, 2, 1]])

x_test = np.array([-0.1, 0, 2])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print X_scaled

print scaler.transform(x_test)