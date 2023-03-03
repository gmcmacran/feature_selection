##################################################################
# Overview
#
# This script compares two feature selection methods.
#
# Method 1: Recursive feature elimination with cross-validation
# Method 2: boruta
#
# Using lightgbm's gpu training. Still takes a loooooong time to 
# run.
##################################################################

# %%
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from lightgbm import LGBMRegressor, LGBMClassifier
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
        approach = "random forest"
        if dataMethod == 1:
            model = LGBMClassifier(n_estimators = 100, max_depth = -1, num_leaves = 1000, 
                                   learning_rate = 1, subsample = 0.63, subsample_freq = 1, 
                                   reg_lambda = 0, reg_alpha = 0, min_child_samples = 1,
                                   colsample_bytree = 1/3, device_type = 'gpu', max_bin =15, n_jobs=8)
        else:
            model = LGBMRegressor(n_estimators = 100, max_depth = -1, num_leaves = 1000, 
                                   learning_rate = 1, subsample = 0.63, subsample_freq = 1, 
                                   reg_lambda = 0, reg_alpha = 0, min_child_samples = 1,
                                   colsample_bytree = 1/3, device_type = 'gpu', max_bin =15, n_jobs=8)
    else:
        approach = "boosting"
        if dataMethod == 1:
            model = LGBMClassifier(boosting_type = 'gbdt', n_estimators = 100, 
                                   device_type = 'gpu', max_bin =15, n_jobs=8)
        else:
            model = LGBMRegressor(boosting_type = 'gbdt', n_estimators = 100, 
                                  device_type = 'gpu', max_bin =15, n_jobs=8)
    
    return model, approach

##############
# run sim
##############
# %%

pieces = []
seed = 60
dataMethod = 2
modelMethod = 1
col = 5
b = 0
for dataMethod in [2]:
    for modelMethod in [1, 2]:
        print("======================")
        print(modelMethod)
        print("======================")
        print("")
        for col in np.arange(5, 105, 10):
            print("col: " + str(col))
            for b in np.arange(0, 3, 1):

                # create data
                seed += 1
                X_train, y_train, dataType = create_data(50000, col, seed, dataMethod)

                baseModel, approach= create_model(modelMethod, dataMethod)

                # RFECV
                model_rfecv = RFECV(estimator = baseModel, n_jobs=1)
                model_rfecv.fit(X_train, y_train)
                n_features_rfecv = model_rfecv.n_features_

                # boruta
                model_boruta = BorutaPy(estimator = baseModel)
                model_boruta.fit(X_train, y_train)
                n_features_boruta = model_boruta.n_features_

                # summarize results into data frame.
                piece = {'model':[approach], 'dataType':[dataType], 'col':[col], 'b':[b], 'rfecv':[n_features_rfecv], 'boruta':[n_features_boruta]}
                piece = pd.DataFrame(piece)
                pieces.append(piece)

                # save progress
                fn = 'data/progress_' + str(seed) + '.csv'
                piece.to_csv(path_or_buf = fn, index=False)

result = pd.concat(pieces)
result.sort_values(['model', 'dataType', 'col', 'b'])

# %%
result.to_csv(path_or_buf = 'data/result.csv', index=False)

