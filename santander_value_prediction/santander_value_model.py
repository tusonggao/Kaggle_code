import pickle
import time
import gc
import hashlib
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def evalue_RMSLE_score(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))

# self-defined eval metric
# f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
# Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_true, y_pred):
    return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False

def convert_2_md5(value):
    return hashlib.md5(str(value).encode('utf-8')).hexdigest()

def write_to_log(*param):
    param_list = [str(s) for s in param]
    log = ' '.join(param_list)
    with open('./outcome/log_file.txt', 'a') as file:
        file.write(log+'\n')
        file.flush()  #立即写入磁盘
        os.fsync(file)  #立即写入磁盘
        

print('santander_value_model prog starting...')

start_t = time.time()
train_df = pd.read_csv('C:/D_Disk/data_competition/Santander_Value_Prediction/data/train.csv', 
                       index_col=0, header=0)
test_df = pd.read_csv('C:/D_Disk/data_competition/Santander_Value_Prediction/data/test.csv',
                      index_col=0, header=0)
print('train_df.shape is {}, test_df.shape is {} load data cost time:{}'.format(
        train_df.shape, test_df.shape, time.time()-start_t))

train_y = train_df['target']
train_X = train_df.drop(['target'], axis=1)
merged_X = train_X.append(test_df)

#merged_X = (merged_X - merged_X.min()) / (merged_X.max() - merged_X.min())
#merged_X = (merged_X - merged_X.mean()) / (merged_X.std())

train_X = merged_X.iloc[:len(train_y)]
test_X = merged_X.iloc[len(train_y):]

X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_X, 
            train_y, test_size=0.2, random_state=42)
print('X_train_new.shape is {}, X_val_new.shape is {}'.format(X_train_new.shape,
      X_val_new.shape))

def test_param(lgbm_param):
    start_t = time.time()
    
    lgbm = lgb.LGBMRegressor(**lgbm_param)
    lgbm.fit(X_train_new, y_train_new, eval_set=[(X_train_new, y_train_new), 
            (X_val_new, y_val_new)], eval_metric=rmsle, 
            verbose=2000, early_stopping_rounds=2000)
    
    best_iteration = lgbm.best_iteration_
    print('best_iteration: {} partial fit cost time: {} '.format(
            best_iteration, time.time()-start_t))
    
    #lgbm.fit(train_X, train_y, eval_set=[(X_train_new, y_train_new), 
    #        (X_val_new, y_val_new)], verbose=2000, early_stopping_rounds=2000)
    #lgbm.fit(X_train_new, y_train_new)
    
    y_predictions_whole = lgbm.predict(train_X)
    RMSLE_score_lgb_whole = round(rmsle(train_y, y_predictions_whole)[1], 5)
        
    y_predictions_train = lgbm.predict(X_train_new)
    RMSLE_score_lgb_train = round(rmsle(y_train_new, y_predictions_train)[1], 5)
    
    y_predictions_val = lgbm.predict(X_val_new)
    RMSLE_score_lgb_val = round(rmsle(y_val_new, y_predictions_val)[1], 5)
    
    len_to_get = int(0.20*len(y_val_new))
    RMSLE_score_lgb_val_20_percent = round(rmsle(y_val_new[:len_to_get],
                                    y_predictions_val[:len_to_get])[1], 5)
    
    print('partial data whole_score: {} train score: {}  test score: {}, '
          'RMSLE_score_lgb_val_20_percent: {} '.format(
           RMSLE_score_lgb_whole, RMSLE_score_lgb_train, 
           RMSLE_score_lgb_val, RMSLE_score_lgb_val_20_percent))
    
    lgbm_param['n_estimators'] = best_iteration
    param_md5_str = convert_2_md5(lgbm_param)
    store_path = 'C:/D_Disk/data_competition/Santander_Value_Prediction/outcome/'
    partial_file_name = '_'.join(['submission_partial', str(RMSLE_score_lgb_val), param_md5_str]) + '.csv'
    full_file_name = '_'.join(['submission_full', str(RMSLE_score_lgb_val), param_md5_str]) + '.csv'
    
    start_t = time.time()
    test_df['target'] = lgbm.predict(test_X)
    test_df['target'].to_csv(store_path+partial_file_name, header=['target'])
    print('get partial predict outcome cost time: ', time.time()-start_t)
    
    start_t = time.time()
    lgbm = lgb.LGBMRegressor(**lgbm_param)
    lgbm.fit(train_X, train_y)
    print('full fit cost time: ', time.time()-start_t)
    
    start_t = time.time()
    test_df['target'] = lgbm.predict(test_X)
    test_df['target'].to_csv(store_path+full_file_name, header=['target'])
    print('get full predict outcome cost time: ', time.time()-start_t)
    
    write_to_log('-'*25, ' md5 value: ', param_md5_str, '-'*25)
    write_to_log('param: ', lgbm_param)
    write_to_log('best_iteration: ', best_iteration)
    write_to_log('valid rmsle: ', RMSLE_score_lgb_val)
    write_to_log('-'*80+'\n')


#lgbm_param = {'n_estimators':50000, 'n_jobs':-1, 'learning_rate':0.1, 
#              'random_state':42, 'max_depth':20, 'min_child_samples':23,
#              'num_leaves':91, 'subsample':0.8, 'colsample_bytree':0.5,
#              'silent':-1, 'verbose':-1}

#test_param(lgbm_param)

lgbm_param = {'n_estimators':90000, 'n_jobs':-1, 'learning_rate':0.15, 
              'random_state':42, 'max_depth':20, 'min_child_samples':17,
              'num_leaves':131, 'subsample':0.8, 'colsample_bytree':0.5,
              'silent':-1, 'verbose':-1}

test_param(lgbm_param)

#lgbm = lgb.LGBMRegressor(n_estimators=50000, n_jobs=-1, learning_rate=0.1, 
#                          random_state=42, max_depth=20, min_child_samples=23,
#                          num_leaves=91, subsample=0.8, colsample_bytree=0.5,
#                          silent=-1, verbose=-1)


