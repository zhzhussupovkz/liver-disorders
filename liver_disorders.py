#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import os
import time
from pandas import DataFrame, read_csv, concat

from sklearn import cross_validation, grid_search
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, roc_auc_score

import pylab as pl
import matplotlib.pyplot as plt

dirs = ['data_plot', 'test_plot']

def start():
    for i in dirs:
        if not os.path.exists(i):
            os.makedirs(i)

def get_analyze_data():
    print "Get analyze data..."
    data = read_csv("./data/bupa.data", sep=',', header=None)
    data['id'] = range(1, len(data)+1)
    label = LabelEncoder()
    label.fit(data[6].drop_duplicates())
    data[6] = label.transform(data[6])
    class_names = list(label.classes_)
    return (data, class_names)

def plot_sgpt_sgot():
    data, class_names = get_analyze_data()

    print "plot sgpt sgot..."
    f = plt.figure(figsize=(8, 6))
    colors = ['r', 'b']
    for k in range(0, 2):
        plt.scatter(data[data[6] == k][2], data[data[6] == k][3], c=colors[k], label="%s" % class_names[k])
    plt.legend()
    f.savefig('./%s/sgpt_sgot_class.png' % dirs[0])

def plot_count_by_class():
    data, class_names = get_analyze_data()
    title = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks']
    for k in range(0, 6):
        data = data.sort([k])

        print 'Plot %s...' % title[k]
        fig, axes = plt.subplots(ncols=1)
        e = data.pivot_table('id', [k], 6, 'count').plot(ax=axes, title='%s' % title[k])
        f = e.get_figure()
        f.savefig('./%s/%s_class.png' % (dirs[0], title[k]))

def plot_box_by_class():
    data, class_names = get_analyze_data()
    title = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks']
    data['class'] = data[6]
    for k in range(0, 6):
        print "plot box %s by class..." % title[k]
        df = concat([data[k], data['class']], axis=1, keys=[title[k], 'class'])
        f = plt.figure(figsize=(8, 6))
        p = df.boxplot(by='class', ax = f.gca())
        f.savefig('./%s/box_%s_class.png' % (dirs[0], title[k]))

def roc_plot():
    data, class_names = get_analyze_data()
    target = data[6]
    train = data.drop(['id', 6], axis = 1)

    model_rfc = RandomForestClassifier(n_estimators = 100, criterion = "entropy", n_jobs = -1)
    model_knc = KNeighborsClassifier(n_neighbors = 10, algorithm = 'brute')
    model_gbc = GradientBoostingClassifier(n_estimators = 100)
    model_lr = LogisticRegression(penalty='l1', tol=0.01)

    ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = cross_validation.train_test_split(train, target, test_size=0.25)

    print 'ROC...'

    pl.clf()
    plt.figure(figsize=(8,6))

    # RandomForestClassifier
    probas = model_rfc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
    fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
    roc_auc  = auc(fpr, tpr)
    pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('RandomForest',roc_auc))

    # GradientBoostingClassifier
    probas = model_gbc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
    fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
    roc_auc  = auc(fpr, tpr)
    pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('GradientBoosting',roc_auc))

    # KNeighborsClassifier
    probas = model_knc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
    fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
    roc_auc  = auc(fpr, tpr)
    pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('KNeighborsClassifier',roc_auc))

    # LogisticRegression
    probas = model_lr.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)
    fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])
    roc_auc  = auc(fpr, tpr)
    pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LogisticRegression',roc_auc))

    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.legend(loc=0, fontsize='small')
    pl.savefig('./%s/roc.png' % dirs[1])

start()
plot_sgpt_sgot()
plot_count_by_class()
plot_box_by_class()
roc_plot()
