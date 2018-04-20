# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 19:04:42 2018

@author: Anumula_Anudeep
"""

from sklearn import datasets
from Gridsearch_multiEstimators import EstimatorSelectionHelper
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target


from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, 
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.svm import SVC

models1 = { 
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC()
}

params1 = { 
    'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
    'RandomForestClassifier': { 'n_estimators': [16, 32] },
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    'SVC': [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    ]
}


helper1 = EstimatorSelectionHelper(models1, params1)
helper1.fit(X_iris, y_iris, scoring='f1', n_jobs=-1)

helper1.score_summary(sort_by='min_score')
