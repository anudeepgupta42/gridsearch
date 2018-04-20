# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 19:08:21 2018

@author: Anumula_Anudeep
"""
from sklearn import datasets
from Gridsearch_multiEstimators import EstimatorSelectionHelper

diabetes = datasets.load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

models2 = { 
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso()
}

params2 = { 
    'LinearRegression': { },
    'Ridge': { 'alpha': [0.1, 1.0] },
    'Lasso': { 'alpha': [0.1, 1.0] }
}


helper2 = EstimatorSelectionHelper(models2, params2)
test = helper2.fit(X_diabetes, y_diabetes, n_jobs=-1)

for k in test.keys:
    print(k, test.grid_searches[k].best_params_, test.grid_searches[k].best_score_)


helper2.score_summary()
