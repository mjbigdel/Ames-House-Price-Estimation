# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 17:32:05 2019

@author: Manoochehr
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression 

# ////////////////////////// import our file for functions ////////////////////////////////////////////////
from DataPreparing import AddPolyFeature, preprocessFunc


# ////////////////////////////////// Read Data Section /////////////////////////////////////////////////////
train_data = pd.read_csv("train.csv", sep=",")
test_data = pd.read_csv("test.csv", sep=",")


# ///////////////////////////// Missing Value Section ////////////////////////////////////////////////////////
train_data,test_data = preprocessFunc(train_data,test_data)


# ///////////////////////////// Outlier Detection Section ////////////////////////////////////////////////////
train_data = train_data.drop(train_data[(train_data['LotFrontage']>300)].index)
train_data = train_data.drop(train_data[(train_data['LotArea']>100000)].index)
train_data.drop(1453, axis=0, inplace=True)
train_data.drop(185, axis=0, inplace=True)
train_data.drop(304, axis=0, inplace=True)
train_data.drop(583, axis=0, inplace=True)
train_data.drop(747, axis=0, inplace=True)
train_data = train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index)
train_data.SalePrice = np.log(train_data.SalePrice)
train_data = train_data[train_data.SalePrice > 10.596660]


# /////////////////////////////// Normalization ////////////////////////////////////////////////////////////
#log transform skewed numeric features > 60%:
from scipy.stats import skew
numeric_feats = train_data.dtypes[train_data.dtypes != "object"].index
skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.60]
skewed_feats = skewed_feats.index
train_data[skewed_feats] = np.log1p(train_data[skewed_feats])
test_data[skewed_feats] = np.log1p(test_data[skewed_feats])


# ///////////////////////////// Feature Engineering ////////////////////////////////////////////////////////
# drop correlated features with corr bigger than 80%
features_corr = ['GarageArea', '1stFlrSF', 'TotRmsAbvGrd','GarageYrBlt','KitchenAbvGr']
train_data.drop(features_corr, axis=1, inplace=True)
test_data.drop(features_corr, axis=1, inplace=True)

# Add some New Features
train_data['NewHouse'] = train_data['YrSold'] - train_data['YearBuilt']
func = lambda x: x['NewHouse'] == 0 and 1.0 or 0.0
train_data['NewHouse'] = train_data.apply(func,axis=1).astype(float)
test_data['NewHouse'] = test_data['YrSold'] - test_data['YearBuilt']
func = lambda x: x['NewHouse'] == 0 and 1.0 or 0.0
test_data['NewHouse'] = test_data.apply(func,axis=1).astype(float)

train_data['OverallSF'] = train_data['2ndFlrSF'] + train_data['TotalBsmtSF']
test_data['OverallSF'] = test_data['2ndFlrSF'] + test_data['TotalBsmtSF']


# ///////////////////////// split target and features /////////////////////////////////////////////////////
train_labels = train_data['SalePrice']
train_data.drop('SalePrice', axis=1, inplace=True)
print(train_labels.describe())


# ///////////////////////// Create dummy Variables of categorical ones. ///////////////////////////////////
Alldata = pd.concat((train_data,test_data),axis =0)
Alldata = pd.get_dummies(Alldata)


# //////////////////////////// split train and test data ////////////////////////////////////////////////
print(train_data.shape[0])
train_data = Alldata.iloc[0:train_data.shape[0]]
test_data = Alldata.iloc[train_data.shape[0]:]


# //////////////////////// Add poly=2 features for 12 most important feature ////////////////////////////
train_data = AddPolyFeature(train_data)
test_data = AddPolyFeature(test_data)


# /////////////////////////////////// Start training /////////////////////////////////////////////////////
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge, LassoCV,LassoLarsCV, ElasticNet


# ////////////////////////////// build a model library /////////////////////////////////////////////////
base_models = [
    LassoCV(alphas = [1, 0.1, 0.001, 0.0005, 0.00001]),	
	LassoLarsCV(),
	ElasticNet(),
    LinearRegression(),
    LinearRegression(),    
    RandomForestRegressor(max_depth=9,
            n_jobs=1, random_state=0,
            n_estimators=8000, max_features=14
        ),
    RandomForestRegressor(max_depth=12,
            n_jobs=1, random_state=0,
            n_estimators=12000, max_features=20	    
        ),
    GradientBoostingRegressor(n_estimators = 8000, max_depth = 6,
                                   min_samples_split = 3, learning_rate = 0.005,
                                   random_state=0, max_features=15, subsample=0.8
        ),
    GradientBoostingRegressor(n_estimators = 10000, max_depth = 8,
                                   min_samples_split = 3, learning_rate = 0.001,
                                   random_state=0, max_features=20, subsample=0.8
        )
    ]


# ////////////////////////////// RMSE function ///////////////////////////////////////////////////////////
def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5

RMSE = make_scorer(mean_squared_error_, greater_is_better=False)


# ///////////////////// Fit base models and predict for test set //////////////////////////////////////////
def fit_predict(train,test,ytr):
    X = train.values
    y = ytr.values
    T = test.values
    folds = KFold(n_splits=10, random_state = 0)
    S_train = np.zeros((X.shape[0],len(base_models)))
    S_test = np.zeros((T.shape[0],len(base_models))) 
    for i,reg in enumerate(base_models):
        print ("Fitting the base model...")
        S_test_i = np.zeros((T.shape[0], folds.n_splits)) 
        for j, (train_idx,test_idx) in enumerate(folds.split(X)):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_holdout = X[test_idx]
            reg.fit(X_train,y_train)
            y_pred = reg.predict(X_holdout)[:]
            S_train[test_idx,i] = y_pred
            S_test_i[:,j] = reg.predict(T)[:]            
        S_test[:,i] = S_test_i.mean(1)
     
    print ("Stacking base models...")
    # tuning the stacker
    param_grid = {
	     'alpha': [1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,0.2,0.3,0.4,0.5,0.8,1e0,3,5,7,1e1],
     }
    
    grid = GridSearchCV(estimator=Ridge(), param_grid=param_grid, n_jobs=1, cv=5, scoring=RMSE)
    grid.fit(S_train, y)
    try:
        print('Param grid:')
        print(param_grid)
        print('Best Params:')
        print(grid.best_params_)
        print('Best CV Score:')
        print(-grid.best_score_)
        print('Best estimator:')
        print(grid.best_estimator_)            
    except:
        pass

    y_pred = grid.predict(S_test)[:]
    return y_pred, -grid.best_score_


#//////////////////////// create file to do submission in Kaggle website //////////////////////////////////////
y_pred, score = fit_predict(train_data,test_data,train_labels)
pred = np.expm1(y_pred)

submission = pd.read_csv("sample_submission.csv")
submission['SalePrice'] = pred
submission.to_csv('submission_2.csv', index=None)
print("success")
