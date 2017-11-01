import sys
import pickle
import pandas as pd
import numpy as np
import random
import math
import mglearn
import sklearn
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

from sklearn import linear_model
from functools import reduce
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from scipy.stats import skew

def convert_object(df):
    obj_df = df.select_dtypes(include=['object']).copy()
    for col_name in obj_df.columns:
        df[col_name] = df[col_name].fillna('NA')

def convert_int64(df):
    int64_df = df.select_dtypes(include=['int64']).copy()
    for col_name in int64_df.columns:
        df[col_name] = df[col_name].fillna(df[col_name].mean())

def convert_float64(df):
    float64_df = df.select_dtypes(include=['float64']).copy()
    for col_name in float64_df.columns:
        df[col_name] = df[col_name].fillna(df[col_name].mean())

def convert_dataframe(df):
    convert_object(df)
    convert_int64(df)
    convert_float64(df)

def rmse_cv(model, X, y):
    rmse= np.sqrt(-cross_val_score(model, X, y, 
                                   scoring="neg_mean_squared_error", cv = 5))
    return rmse

def log_transform(df):
    numeric_feats = df.dtypes[df.dtypes != "object"].index
    skewed_features = df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_features = skewed_features[skewed_features > 0.75]
    skewed_features = skewed_features.index
    print('skewed_features is ', skewed_features)
    df[skewed_features] = np.log1p(df[skewed_features])

if __name__=='__main__':
    start_time = time.time()

    train_data_frame = pd.read_csv('./data/train.csv', index_col=['Id'])
    test_data_frame = pd.read_csv('./data/test.csv', index_col=['Id'])
    
    print('train_data_frame shape', train_data_frame.shape)
    
#    train_data_frame['SalePrice'].value_counts().plot(kind='barh')
    
    X_train = train_data_frame.drop(['SalePrice'], axis=1)
    X_test = test_data_frame
    y_train = np.log1p(train_data_frame['SalePrice'])
    
    concated_df = pd.concat([X_train, X_test])
    log_transform(concated_df)
    concated_df = pd.get_dummies(concated_df)
    concated_df = concated_df.fillna(concated_df.mean())
    
    X_train = concated_df[:len(X_train)]
    X_test = concated_df[len(X_train):]
    print('X_train shape is ', X_train.shape)
    print('train_data_frame shape is ', train_data_frame.shape)
    print('training start...')

#    X_train, X_test, y_train, y_test = train_test_split(
#                                            train_data_frame[train_cols], 
#                                            train_data_frame['SalePrice'], 
#                                            random_state=42)
######################################################################

    model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y_train)
    print('rmse_cv(model_lasso).mean() is ', 
          rmse_cv(model_lasso, X_train, y_train).mean())
    coef = pd.Series(model_lasso.coef_, index = X_train.columns)
    print("Lasso picked " + str(sum(coef != 0)) + 
          " variables and eliminated the other " +  
           str(sum(coef == 0)) + " variables")
    lasso_predictions = np.expm1(model_lasso.predict(X_test))

######################################################################
                                            
#    best_ratio = 0
#    best_score = -1000
#    scores_mean_list = []
#    ratio_list = []
#    for ratio in range(10, 100, 10):
#        print('ratio is ', ratio)
#        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
#        rf = RandomForestRegressor(n_estimators=1000,
#                               max_features=int(len(train_cols)*ratio/100),
#                               max_depth=4,                               
#                               n_jobs=4)
#        scores = cross_val_score(rf, train_data_frame[train_cols], 
#                                 train_data_frame['SalePrice'], 
#                                 cv=kfold)
#        print('ratio is ', ratio, 'scores mean is ', scores.mean())
#        scores_mean_list.append(scores.mean())
#        ratio_list.append(ratio)
#        if best_score < scores.mean():
#            best_score = scores.mean()
#            best_ratio = ratio

#    rf = RandomForestRegressor(n_estimators=3000, max_features=0.5, 
#                               oob_score=True, random_state=17, 
#                               n_jobs=2)
#    rf = rf.fit(X_train, y_train)
#    with open('./dumps/rf_regression.pkl', 'wb') as f:
#        pickle.dump(rf, f)    
    pickle_in = open('./dumps/rf_regression.pkl', 'rb')
    rf = pickle.load(pickle_in)
    print("rf accuracy on training set:", rf.score(X_train, y_train))
    predictions0 = np.expm1(rf.predict(X_test))

###########################################################################

#gbrt grid_search.best_params_ is  {'max_depth': 4, 'n_estimators': 1500, 'learning_rate': 0.05}
#gbrt grid_search.best_score_ is  0.904895358041

#    param_grid = {'n_estimators': [1500],
#                  'learning_rate': [0.005, 0.01, 0.05, 0.1],
#                  'max_depth': [2, 4, 6, 8]}
#                  
#    grid_search = GridSearchCV(GradientBoostingRegressor(random_state=17), 
#                               param_grid, cv=5, n_jobs=3)
#    grid_search.fit(X_train, y_train)
#    
#    with open('./dumps/gbrt_regression.pkl', 'wb') as f:
#        pickle.dump(grid_search, f)    
    pickle_in = open('./dumps/gbrt_regression.pkl', 'rb')
    grid_search = pickle.load(pickle_in)
    
    test_score = grid_search.score(X_train, y_train)
    print('GradientBoostingRegressor grid_search best score is ', test_score)
    gbrt_predictions = np.expm1(grid_search.predict(X_test))
    
    print('gbrt grid_search.best_params_ is ', grid_search.best_params_)
    print('gbrt grid_search.best_score_ is ', grid_search.best_score_)
    
#    gbr = GradientBoostingRegressor(n_estimators=1500, max_depth=2, 
#                                    learning_rate=0.1, random_state=17)
#    scores = cross_val_score(gbr, X_train, y_train, cv=5)
#    print('scores mean is ', scores.mean())
#    gbr.fit(X_train, y_train)
#    
#    with open('./dumps/gbrt_regression.pkl', 'wb') as f:
#        pickle.dump(gbr, f)    
#    pickle_in = open('./dumps/gbrt_regression.pkl', 'rb')
#    gbr = pickle.load(pickle_in)
#    
#    print("gbr accuracy on training set:", gbr.score(X_train, y_train))
#    predictions1 = np.expm1(gbr.predict(X_test))
    
#######################################################################
    
#    param_grid = {'kernel': ["rbf"],
#                  'C'     : np.logspace(-5, 5, num=11, base=10.0),
#                  'gamma' : np.logspace(-5, 5, num=11, base=10.0)}
                  
    df_concated = pd.concat([X_train, X_test])
    scaler = StandardScaler().fit(df_concated)
    df_concated_scaled = scaler.transform(df_concated)
    X_train_scaled = df_concated_scaled[:len(X_train)]
    X_test_scaled = df_concated_scaled[len(X_train):]
    
#    grid_search = GridSearchCV(svm.SVR(), param_grid, cv=5, n_jobs=2)
#    grid_search.fit(X_train_scaled, y_train)
#    
#    with open('./dumps/svr_regression.pkl', 'wb') as f:
#        pickle.dump(grid_search, f)    
    pickle_in = open('./dumps/svr_regression.pkl', 'rb')
    grid_search = pickle.load(pickle_in)
    
    test_score = grid_search.score(X_train_scaled, y_train)
    print('svr rbf best score is ', test_score)
    
    svr_predictions = np.expm1(grid_search.predict(X_test_scaled))
    print('svr rbf grid_search.best_params_ is ', grid_search.best_params_)
    print('svr rbf grid_search.best_score_ is ', grid_search.best_score_)

########################################################################

#    param_grid = {'kernel': ["sigmoid"],
#                  'C'     : np.logspace(-5, 5, num=11, base=10.0),
#                  'gamma' : np.logspace(-5, 5, num=11, base=10.0)}
#                  
#    df_concated = pd.concat([X_train, X_test])
#    scaler = StandardScaler().fit(df_concated)
#    df_concated_scaled = scaler.transform(df_concated)
#    X_train_scaled = df_concated_scaled[:len(X_train)]
#    X_test_scaled = df_concated_scaled[len(X_train):]
#    
#    grid_search = GridSearchCV(svm.SVR(), param_grid, cv=5, n_jobs=2)
#    grid_search.fit(X_train_scaled, y_train)
#    
#    with open('./dumps/svr_sigmoid_regression.pkl', 'wb') as f:
#        pickle.dump(grid_search, f)    
#    pickle_in = open('./dumps/svr_sigmoid_regression.pkl', 'rb')
#    grid_search = pickle.load(pickle_in)
#    
#    predictions3 = np.expm1(grid_search.predict(X_test_scaled))
#    print('svr sigmoid grid_search.best_params_ is ', grid_search.best_params_)
#    print('svr sigmoid grid_search.best_score_ is ', grid_search.best_score_)

#######################################################################
#svr grid_search.best_params_ is  {'C': 100.0, 'gamma': 1.0000000000000001e-05, 'kernel': 'linear'}
#svr grid_search.best_score_ is  0.845945545511
#time cost is  28016.53100013733


#    param_grid = {'kernel': ["linear"],
#                  'C'     : np.logspace(-5, 5, num=11, base=10.0),
#                  'gamma' : np.logspace(-5, 5, num=11, base=10.0)}
#                  
#    df_concated = pd.concat([X_train, X_test])
#    scaler = StandardScaler().fit(df_concated)
#    df_concated_scaled = scaler.transform(df_concated)
#    X_train_scaled = df_concated_scaled[:len(X_train)]
#    X_test_scaled = df_concated_scaled[len(X_train):]
#    
#    grid_search = GridSearchCV(svm.SVR(), param_grid, cv=5, n_jobs=2)
#    grid_search.fit(X_train_scaled, y_train)
    
#    with open('./dumps/svr_linear_regression.pkl', 'wb') as f:
#        pickle.dump(grid_search, f)    
#    pickle_in = open('./dumps/svr_linear_regression.pkl', 'rb')
#    grid_search = pickle.load(pickle_in)
#    
#    predictions4 = grid_search.predict(X_test_scaled)
#    print('svr grid_search.best_params_ is ', grid_search.best_params_)
#    print('svr grid_search.best_score_ is ', grid_search.best_score_)

#    predictions = (predictions0 + predictions1 + 3*predictions2)/5.0
    predictions = lasso_predictions
    predictions = gbrt_predictions
    predictions = svr_predictions
    predictions = 0.7*lasso_predictions + 0.15*gbrt_predictions + 0.15*svr_predictions

#######################################################################

    outcome_data_frame = pd.DataFrame(
                              data = predictions,
                              columns= ['SalePrice'],
                              index=X_test.index.values)
    
    outcome_data_frame.to_csv('./data/submission.csv', index_label='Id')

    end_time = time.time()    
    print('time cost is ', end_time-start_time)
    
########################################################################
    
#    print(outcome_data_frame.shape)
#    print(lr.predict(test_data_frame[train_cols]))

    
#    data_frame['MSSubClass'].hist()
#    data_frame['MSZoning'].hist()
#    ms_count = 
#    data_frame['MSZoning'].value_counts().plot(kind='barh')
#    data_frame['LotFrontage'].value_counts().plot(kind='barh')
#    data_frame['LotArea'].hist()
#    data_frame['LotArea'].value_counts().plot(kind='barh')
#    data_frame['LotFrontage'].hist()
#    data_frame['LotArea'].hist()
#    data_frame['Street'].hist()
#    data_frame['Alley'].hist()
#    data_frame['LotShape'].hist()

#    data_frame['MSZoning'].hist()
#    data_frame['MSZoning'].hist()
#    data_frame['MSZoning'].hist()
#    data_frame['MSZoning'].hist()
#    data_frame['MSZoning'].hist()
#    data_frame['MSZoning'].hist()
#    data_frame['MSZoning'].hist()
#    data_frame['MSZoning'].hist()
#    data_frame['MSZoning'].hist()
