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

from product_analysis import findCommonFeatures

class GenderPredictor(object):

    def __init__(self):
        self._loadTrainingData()

    def _loadTrainingLabels(self):
        genders = []
        with open('data/trainingLabels.csv') as f:
            for line in f.readlines():
                g = line.strip()
                if g == 'male':
                    genders.append(1)
                else:
                    genders.append(0)
        return genders

    def _loadTrainingData(self):
        genders = self._loadTrainingLabels()
        self.test_data = []
        with open('data/trainingData.csv') as f:
            line_count = -1
            for line in f.readlines():
                line_count += 1
                d = line.strip().split(',')
                products_str = d[3]
                ids = products_str.split(';')
                products = []
                for product in ids:
                    ids = product.split('/')[0:4]
                    products.append((ids[0], ids[1], ids[2], ids[3]))
                x = {'gender': genders[line_count],
                     'start_date': d[1],
                     'end_date': d[2],
                     'products': products}
                self.test_data.append(x)

        self.A_set, self.A_index = findCommonFeatures(0)
        self.B_set, self.B_index = findCommonFeatures(1)
        self.C_set, self.C_index = findCommonFeatures(2)

    def setParameters(self, min_rule_acc=[0.1, 0.95], female_ratio=1.0, n_product_least=1):
        self.min_rule_acc = min_rule_acc
        self.female_ratio = female_ratio
        self.n_product_least = n_product_least
        self.feature_selection = False
        self.selected_feature_idx = None

    def _loadRules(self):
        self.rules = {}
        with open('data/rules_no_dup.txt') as f:
            for line in f.readlines():
                d = line.split()
                name = d[0]
                conf = int(d[1])
                acc = float(d[2])
                if float(d[2]) > 0.5:
                    label = 0
                else:
                    label = 1
                if (acc >= max(self.min_rule_acc) or acc <= min(self.min_rule_acc)) and conf >= 10:
                    self.rules[name] = label

    def _judgeByRules(self, x):
        results = set()
        for product in x['products']:
            for name in product:
                res = self.rules.get(name, -1)
                if res != -1:
                    results.add(res)
        if len(results) == 1:
            return list(results)[0]
        return -1


    def extractFeatures(self, x, feature_idx=None):
        # start hour
        st_time = datetime.strptime(x['start_date'], '%Y-%m-%d %H:%M:%S')
        time_feature = [0 for i in xrange(24)]
        time_feature[st_time.hour] = 1

        # total_count of viewed products is not useful
        total_view = len(x['products'])
        A_feature = [0 for i in xrange(len(self.A_index))]
        B_feature = [0 for i in xrange(1 + len(self.B_index))]
        C_feature = [0 for i in xrange(1 + len(self.C_index))]
        for product in x['products']:
            A_index = self.A_index[product[0]]
            A_feature[A_index] += 1.0 / total_view

            if product[1] in self.B_set:
                B_index = self.B_index[product[1]]
            else:
                B_index = len(B_feature) - 1
            B_feature[B_index] += 1.0 / total_view


            if product[2] in self.C_set:
                C_index = self.C_index[product[2]]
            else:
                C_index = len(C_feature) - 1
            C_feature[C_index] += 1.0 / total_view

        raw_feature = time_feature + A_feature + B_feature + C_feature + [total_view]
        if feature_idx is None:
            return raw_feature

        selected_feature = []
        for id in feature_idx:
            selected_feature.append(raw_feature[id])
        return selected_feature

    def trainModel(self, model, retrain=False):
        if model == 'GBDT':
            self.predictor = GradientBoostingRegressor(n_estimators=100, max_depth=3, subsample=0.8)
        elif model == 'RF':
            self.predictor = RandomForestClassifier()
        elif model == 'LR':
            self.predictor = LogisticRegression()
        elif model == 'SVM':
            self.predictor = svm.SVC()
        elif model == 'NB':
            self.predictor = BernoulliNB()
        elif model == 'DT':
            self.predictor = DecisionTreeClassifier()
        elif model == 'KNN':
            self.predictor = KNeighborsClassifier(n_neighbors=2)
        else:
            assert False

        features = []
        labels = []
        for d in self.training_data:
            features.append(self.extractFeatures(d))
            labels.append(d['gender'])

        # if self.feature_selection:
        #     selector = GradientBoostingRegressor(n_estimators=100, max_depth=3, subsample=0.8)
        #     selector.fit(features, labels)
        #     self.selected_feature_idx = []
        #     for i in xrange(len(selector.feature_importances_)):
        #         if selector.feature_importances_[i] > 0.000001:
        #             self.selected_feature_idx.append(i)
        #     print len(self.selected_feature_idx)
        #
        #     features = []
        #     labels = []
        #     for d in self.training_data:
        #         features.append(self.extractFeatures(d, self.selected_feature_idx))
        #         labels.append(d['gender'])

        features = np.array(features)

        if model == 'LR':
            self.standardize = True
        else:
            self.standardize = False

        if self.standardize:
            self._scaler = StandardScaler()
            features = self._scaler.fit_transform(features)

        self.predictor.fit(features, labels)
        # print len(features[0])

        # if retrain:
        #     selected_male = []
        #     selected_female = []
        #     for i in xrange(len(labels)):
        #         feature = features[i]
        #         true_y = labels[i]
        #         predict_y = self.predictor.predict(feature)
        #         if true_y == 1:
        #             selected_male.append((true_y - predict_y, true_y, feature))
        #         else:
        #             selected_female.append((predict_y, true_y, feature))
        #
        #     selected_male = sorted(selected_male, key=lambda x: x[0])
        #     selected_female = sorted(selected_female, key=lambda x: x[0])
        #
        #     n_discard_male = min(200, int(len(selected_male) * 0.9))
        #     n_discard_female = min(200, int(len(selected_female) * 0.9))
        #
        #     print 'After re-selection, %d female and %d male' % (len(selected_male) - n_discard_male, len(selected_female) - n_discard_female)
        #     features = []
        #     labels = []
        #     for d in selected_male[0: len(selected_male) - n_discard_male] + selected_female[0: len(selected_female) - n_discard_female]:
        #         labels.append(d[1])
        #         features.append(d[2])
        #
        #     self.predictor.fit(features, labels)


    def predict(self, d, use_rule=True):
        feature = self.extractFeatures(d, self.selected_feature_idx)
        # print len(feature)
        if self.standardize:
            feature = self._scaler.transform(feature)
        predicted_y_by_classifier = int(self.predictor.predict(feature) + 0.5)
        predicted_y_by_rules = self._judgeByRules(d)
        if use_rule and predicted_y_by_rules !=- 1:
            final_label = predicted_y_by_rules
        else:
            final_label = predicted_y_by_classifier

        # if true_label is not None and true_label != final_label:
        #     print true_label, predicted_y_by_rules, predicted_y_by_classifier
        #     pprint(d)
        #     print
        return final_label

    def testByClassifier(self, test_data):

        total_male = 0
        correct_predicted_male = 0

        total_female = 0
        correct_predicted_female = 0

        predicted_as_male = 0
        predicted_as_female = 0

        for d in test_data:
            true_label = d['gender']
            final_predicted_label = self.predict(d)

            if final_predicted_label == 1:
                predicted_as_male += 1
            else:
                predicted_as_female += 1

            if true_label == 0:
                total_female += 1
                if final_predicted_label == 0:
                    correct_predicted_female += 1
            elif true_label == 1:
                total_male += 1
                if final_predicted_label == 1:
                    correct_predicted_male += 1
            else:
                assert False

            # if final_predicted_label != true_label:
            #     # print true_label, final_predicted_label
            #     pprint(d)

        acc_male = correct_predicted_male * 1.0 / total_male
        acc_female = correct_predicted_female * 1.0 / total_female
        print acc_male, acc_female, (acc_female + acc_male) / 2
        print predicted_as_male, total_male, predicted_as_female, total_female
        return (acc_female + acc_male) / 2

    def splitTrainingAndEvaluationData(self, training_proportion=1.0):
        random.shuffle(self.test_data)

        n_training = int(0.5 + len(self.test_data) * training_proportion)
        self.all_training_data = self.test_data[0:n_training]
        self.evaluation_data = self.test_data[n_training:]

    def downsampleTrainingData(self):
        female_data = []
        male_data = []
        for d in self.training_data:
            if d['gender'] == 0:
                female_data.append(d)
            else:
                male_data.append(d)
        random.shuffle(female_data)
        female_data = female_data[0:int(0.5 + self.female_ratio * len(male_data))]
        # print 'After downsample female data, there are %d females and %d males' % (len(female_data), len(male_data))

        self.training_data = female_data + male_data

    def trainingDataSelection(self):
        # this should be run before _downsampleTrainingData
        self.training_data = []
        for d in self.all_training_data:
            if len(d['products']) >= self.n_product_least or d['gender'] == 1:
                self.training_data.append(copy.deepcopy(d))

def outputTestData():
    test_data = []
    with open('data/testData.csv') as f:
        for line in f.readlines():
            d = line.strip().split(',')
            products_str = d[3]
            ids = products_str.split(';')
            products = []
            for product in ids:
                ids = product.split('/')[0:4]
                products.append((ids[0], ids[1], ids[2], ids[3]))
            x = {'start_date': d[1],
                 'end_date': d[2],
                 'products': products}
            test_data.append(x)

    gp = GenderPredictor()
    gp.splitTrainingAndEvaluationData(1.0)
    gp.setParameters(min_rule_acc=[0.1, 0.95], female_ratio=1)
    gp._loadRules()

    gp.trainingDataSelection()
    gp.downsampleTrainingData()
    gp.trainModel('GBDT')

    with open('testLabels.txt', 'w') as f:
        for d in test_data:
            predicted_label = gp.predict(d)
            if predicted_label == 0:
                f.write('female\n')
            else:
                f.write('male\n')


def tune():
    gp = GenderPredictor()
    gp.splitTrainingAndEvaluationData(training_proportion=0.8)

    best_acc = 0
    best_paras = []

    # min_rule_acc=0.95, female_ratio=1.0, n_product_least
    for min_rule_acc in [[0.3, 0.9], [0.2, 0.95], [0.1, 0.95]]:
        for female_ratio in [0.9, 1.0, 1.1]:
            for n_product_least in [1, 2]:
                gp.setParameters(min_rule_acc=min_rule_acc,
                                 female_ratio=female_ratio,
                                 n_product_least=n_product_least)
                gp._loadRules()
                gp.trainingDataSelection()
                gp.downsampleTrainingData()
                classifiers = ['LR', 'GBDT']
                for model in classifiers:
                    # print model, min_rule_acc, female_ratio, n_product_least

                    gp.trainModel(model)
                    acc = gp.testByClassifier(gp.evaluation_data)
                    if acc > best_acc:
                        best_acc = acc
                        best_paras = [model, min_rule_acc, female_ratio, n_product_least]

    print best_acc, best_paras


def singleTune():
    gp = GenderPredictor()
    gp.splitTrainingAndEvaluationData(training_proportion=0.8)

    gp.setParameters(min_rule_acc=[0.1, 0.95])
    gp._loadRules()
    gp.trainingDataSelection()
    gp.downsampleTrainingData()
    classifiers = ['LR']
    for model in classifiers:
        # print model, min_rule_acc, female_ratio, n_product_least

        gp.trainModel(model)
        gp.testByClassifier(gp.evaluation_data)
        # if model == 'LR':
        #     print gp.predictor.coef_

# {'male': 3297, 'female': 11703}
if __name__ == '__main__':
    # outputTestData()
    # for i in xrange(100):
    #     tune()
    singleTune()


