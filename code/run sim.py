##################################################################
# Overview
#
# This script compares two feature selection methods.
#
# Method 1: Recursive feature elimination with cross-validation
# Method 2: boruta
#
# Takes a loooooong time to run.
##################################################################

# %%
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from boruta import BorutaPy

##############
# Set working directory
##############
# %%
os.chdir('S:/Python/projects/feature_selection')

##############
# Define helpers
##############
# %%
def create_data(nrow, col, seed, dataMethod):
    if dataMethod == 1:
        X, y = make_classification(n_samples = nrow,  n_features = 100, n_informative= col, n_redundant = 100-col, random_state=seed)
        type = 'classification'
    else:
        X, y = make_regression(n_samples = nrow,  n_features = 100, n_informative= col, random_state=seed)
        type = 'regression'
    
    return X, y, type

def create_model(modelMethod, dataMethod):
    if modelMethod == 1:
        approach = "extra trees"
        if dataMethod == 1:
            model = ExtraTreesClassifier(n_jobs=-1)
        else:
            model = ExtraTreesRegressor(n_jobs=-1)
    elif modelMethod == 2:
        approach = "random forest"
        if dataMethod == 1:
            model = RandomForestClassifier(n_jobs=-1)
        else:
            model = RandomForestRegressor(n_jobs=-1)
    else:
        approach = "boosting"
        if dataMethod == 1:
            model = GradientBoostingClassifier()
        else:
            model = GradientBoostingRegressor()
    
    return model, approach

##############
# run sim
##############
# %%
pieces = []
seed = 0
for dataMethod in [1, 2]:
    for modelMethod in [1, 2, 3]:
        print(modelMethod)
        for col in np.arange(5, 105, 5):
            for b in np.arange(0, 10, 1):

                # create data
                seed += 1
                X_train, y_train, dataType = create_data(100000, col, seed, dataMethod)

                baseModel, approach= create_model(modelMethod, dataMethod)

                # RFECV
                model_rfecv = RFECV(estimator = baseModel)
                model_rfecv.fit(X_train, y_train)
                n_features_rfecv = model_rfecv.n_features_

                # boruta
                model_boruta = BorutaPy(estimator = baseModel, n_estimators='auto')
                model_boruta.fit(X_train, y_train)
                n_features_boruta = model_boruta.n_features_

                # summarize results into data frame.
                piece = {'model':[approach], 'dataType':[dataType], 'col':[col], 'b':[b], 'rfecv':[n_features_rfecv], 'boruta':[n_features_boruta]}
                piece = pd.DataFrame(piece)
                pieces.append(piece)

result = pd.concat(pieces)
result.sort_values(['model', 'dataType', 'col', 'b'])

# %%
result.to_csv(path_or_buf = 'data/result.csv', index=False)

# %%
