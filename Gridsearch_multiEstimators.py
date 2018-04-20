import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV

class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
    
    def fit(self, X, y, cv=3, n_jobs=1, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, 
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            self.grid_searches[key] = gs    
        return self
    def score_summary(self, sort_by='min_score'):
        
        def row(key, gsc):
            d = {
                 'estimator': key,
                 'min_score': gsc.best_score_,
                 'max_score': gsc.best_params_,
                 #'mean_score': np.mean(scores),
                 #'std_score': np.std(scores),
            }
            return pd.Series( d.items())
                      
        rows = [row(k, gsc) 
                     for k in self.keys
                     for gsc in self.grid_searches[k]]
        df = pd.concat(rows, axis=1).T.sort([sort_by], ascending=False)
        
        columns = ['estimator', 'min_score', 'mean_score']
        columns = columns + [c for c in df.columns if c not in columns]
        
        return df[columns]
