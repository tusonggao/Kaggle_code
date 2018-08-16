import pickle
import time
import gc
import os
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

def convert_2_md5(value):
    return hashlib.md5(str(value).encode('utf-8')).hexdigest()

def write_to_log(*param):
    param_list = [str(s) for s in param]
    log = ' '.join(param_list)
    with open('./outcome/log_file.txt', 'a') as file:
        file.write(log+'\n')
        file.flush()  #立即写入磁盘
        os.fsync(file)  #立即写入磁盘

# RMSE(root-mean-square error)
# self-defined eval metric
# f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
# Root Mean Squared Logarithmic Error (RMSLE)
def rmse(y_true, y_pred):
    y_pred = np.where(y_pred>0, y_pred, 0)
    return 'RMSE', np.sqrt(mean_squared_error(y_true, y_pred)), False

def rmse_new(y_true, y_pred):
    if np.sum(y_pred<0)>0:
        print('negative exits.', np.sum(y_pred<0))
    else:
        print('negative not exits.')
    y_pred = np.where(y_pred>0, y_pred, 0)
    return 'RMSE', np.sqrt(mean_squared_error(y_true, y_pred)), False

#print('prog starting, hello world!')
start_t = time.time()
merged_df = pd.read_csv('C:/D_Disk/data_competition/gamer_value/data/merged_df.csv', 
                       index_col=0, header=0)
print('merged_df.shape is {} load data cost time:{}'.format(
        merged_df.shape, time.time()-start_t))

train_df = merged_df[merged_df['prediction_pay_price']!=-99999]
train_y = train_df['prediction_pay_price'].values
train_X = train_df.drop(['prediction_pay_price'], axis=1).values

test_df = merged_df[merged_df['prediction_pay_price']==-99999]
test_X = test_df.drop(['prediction_pay_price'], axis=1).values

#del merged_df, train_df, test_df
#gc.collect()

X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_X, 
            train_y, test_size=0.25, random_state=42)
print('X_train_new.shape is {}, X_val_new.shape is {}, test_X.shape is {}'.format(
      X_train_new.shape, X_val_new.shape, test_X.shape))


#lgbm = lgb.LGBMRegressor(n_estimators=5000, n_jobs=-1, learning_rate=0.08, 
#                          random_state=42, max_depth=13, min_child_samples=400,
#                          num_leaves=700, subsample=0.7, colsample_bytree=0.85,
#                          silent=-1, verbose=-1)


#lgbm = lgb.LGBMRegressor(n_estimators=5000, n_jobs=-1, learning_rate=0.05, 
#                          random_state=42, max_depth=7, min_child_samples=150,
#                          num_leaves=3000, subsample=0.7, colsample_bytree=0.8,
#                          silent=-1, verbose=-1)

#lgbm = lgb.LGBMRegressor(n_estimators=6000, n_jobs=-1, learning_rate=0.08, 
#                          random_state=42, max_depth=13, min_child_samples=800,
#                          num_leaves=151, subsample=0.8, colsample_bytree=0.9,
#                          boosting_type='dart', reg_alpha=0.1, reg_lambda=0.05,
#                          silent=-1, verbose=-1)

#lgbm.fit(X_train_new, y_train_new, eval_set=[(X_val_new, y_val_new)], 
#         eval_metric=rmse, verbose=200, early_stopping_rounds=600)

#lgbm.fit(train_X, train_y, eval_set=[(X_train_new, y_train_new), 
#        (X_val_new, y_val_new)], eval_metric=rmse, 
#        verbose=200, early_stopping_rounds=500)
    
#lgbm.fit(X_train_new, y_train_new)

def test_param(lgbm_param):
    lgbm = lgb.LGBMRegressor(**lgbm_param)
    lgbm.fit(X_train_new, y_train_new, eval_set=[(X_train_new, y_train_new), 
            (X_val_new, y_val_new)], eval_metric=rmse, 
            verbose=200, early_stopping_rounds=600)
    
    best_iteration = lgbm.best_iteration_
    y_predictions_whole = lgbm.predict(train_X)
    RMSLE_score_lgb_whole = round(rmse(train_y, y_predictions_whole)[1], 4)
        
    y_predictions_train = lgbm.predict(X_train_new)
    RMSLE_score_lgb_train = round(rmse(y_train_new, y_predictions_train)[1], 4)
    
    y_predictions_val = lgbm.predict(X_val_new)
    RMSLE_score_lgb_val = round(rmse(y_val_new, y_predictions_val)[1], 4)
    RMSLE_score_lgb_val_new = round(rmse_new(y_val_new, y_predictions_val)[1], 4)
#    RMSLE_score_lgb_val = rmse(y_val_new, y_predictions_val)[1]
#    RMSLE_score_lgb_val_new = rmse_new(y_val_new, y_predictions_val)[1]
    
    len_to_get = int(0.20*len(y_val_new))
    RMSLE_score_lgb_val_20_percent = rmse(y_val_new[:len_to_get], y_predictions_val[:len_to_get])[1]
    
    print('partial data whole_score: {} train score: {}  test score: {}, '
          'test score new: {}, RMSLE_score_lgb_val_20_percent: {}'.format(
           RMSLE_score_lgb_whole, RMSLE_score_lgb_train, RMSLE_score_lgb_val,
           RMSLE_score_lgb_val_new, RMSLE_score_lgb_val_20_percent))
    
    start_t = time.time()
    prediction_pay_price_ss = lgbm.predict(test_X)
    prediction_pay_price_ss = np.round(prediction_pay_price_ss, 5)
    prediction_pay_price_ss = np.where(prediction_pay_price_ss>0, 
                                       prediction_pay_price_ss, 0)
    test_df['prediction_pay_price'] = prediction_pay_price_ss
    
    lgbm_param['n_estimators'] = best_iteration
    param_md5_str = convert_2_md5(lgbm_param)
    store_path = 'C:/D_Disk/data_competition/gamer_value/outcome/'
    partial_file_name = '_'.join(['submission_partial', str(RMSLE_score_lgb_val), param_md5_str]) + '.csv'
    full_file_name = '_'.join(['submission_full', str(RMSLE_score_lgb_val), param_md5_str]) + '.csv'
    
    test_df['prediction_pay_price'].to_csv(store_path+partial_file_name,
           header=['prediction_pay_price'])
    
    print('partial get predict outcome cost time: ', time.time()-start_t)
    
    start_t = time.time()
    lgbm = lgb.LGBMRegressor(**lgbm_param)
    lgbm.fit(train_X, train_y)
    print('full fit cost time: ', time.time()-start_t)
    
    start_t = time.time()
    prediction_pay_price_ss = lgbm.predict(test_X)
    prediction_pay_price_ss = np.round(prediction_pay_price_ss, 5)
    prediction_pay_price_ss = np.where(prediction_pay_price_ss>0, 
                                       prediction_pay_price_ss, 0)
    test_df['prediction_pay_price'] = prediction_pay_price_ss
    test_df['prediction_pay_price'].to_csv(store_path+full_file_name,
           header=['prediction_pay_price'])
    
    print('full predict cost time: ', time.time()-start_t)
    
    write_to_log('-'*25, ' md5 value: ', param_md5_str, '-'*25)
    write_to_log('param: ', lgbm_param)
    write_to_log('best_iteration: ', best_iteration)
    write_to_log('valid rmse: ', RMSLE_score_lgb_val)
    write_to_log('-'*80+'\n')

#lgbm_param = {'n_estimators':5000, 'n_jobs':-1, 'learning_rate':0.05, 
#              'random_state':42, 'max_depth':8, 'min_child_samples':30,
#              'num_leaves':1000, 'subsample':0.7, 'colsample_bytree':0.85,
#              'reg_alpha':0.1, 'reg_lambda':0.05, 'silent':-1, 'verbose':-1}
#
#test_param(lgbm_param)
    
lgbm_param = {'n_estimators':5000, 'n_jobs':-1, 'learning_rate':0.05, 
              'random_state':42, 'max_depth':7, 'min_child_samples':21,
              'num_leaves':3000, 'subsample':0.7, 'colsample_bytree':0.85,
              'silent':-1, 'verbose':-1}

test_param(lgbm_param)

#lgbm_param = {'n_estimators':5000, 'n_jobs':-1, 'learning_rate':0.05, 
#              'random_state':42, 'max_depth':9, 'min_child_samples':10,
#              'num_leaves':3000, 'subsample':0.7, 'colsample_bytree':0.85,
#              'reg_alpha':0.05, 'reg_lambda':0.05, 'silent':-1, 'verbose':-1}
#
#test_param(lgbm_param)


