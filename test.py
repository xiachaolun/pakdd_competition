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
#
# from sklearn.preprocessing import StandardScaler
#
# X = np.array([[0.1, 0.5, 0],
#               [0.3, -0.4, 4],
#               [-0.3, 2, 1]])
#
# x_test = np.array([-0.1, 0, 2])
#
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# print X_scaled
#
# print scaler.transform(x_test)

# import matplotlib.pyplot as plt
#
# import networkx as nx
#
# G=nx.Graph()
#
# G.add_node(1)
# G.add_node(2)
#
# G.add_edge(1, 2, weight=1.0)
#
# print G.has_edge(1, 2)
# G[1][2]['weight'] = G[1][2]['weight'] + 2
# print G[2][1]

from multiprocessing.pool import ThreadPool



