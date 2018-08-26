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

def convert_2_md5(value):
    return hashlib.md5(str(value).encode('utf-8')).hexdigest()

def write_to_log(*param):
    param_list = [str(s) for s in param]
    log = ' '.join(param_list)
    with open('C:/D_Disk/data_competition/home_credit'
              '/outcome/log_file.txt', 'a') as file:
        file.write(log+'\n')
        file.flush()  #立即写入磁盘
        os.fsync(file)  #立即写入磁盘

best_dtypes_df = pd.read_csv(
    'C:/D_Disk/data_competition/home_credit/data/processed/best_dtypes.csv',
    index_col=0)
dtypes_dict = dict(best_dtypes_df['best_dtype'])

start_t = time.time()
data_dir_path = 'C:/D_Disk/data_competition/home_credit/data/'
merged_df = pd.read_csv(data_dir_path + '/processed/merged_df.csv',
                        dtype=dtypes_dict, index_col=0)
print('previous merged_df.shape is ', merged_df.shape,
      'read cost time: ', time.time()-start_t)
#print('merged_df info', merged_df.info(memory_usage='deep'))

start_t = time.time()
merged_df = pd.get_dummies(merged_df)
print('after get dummies, merged_df.shape is ', merged_df.shape,
      'get dummies cost time: ', time.time()-start_t)

merged_df_train = merged_df[pd.notna(merged_df['TARGET'])]
merged_df_test = merged_df[pd.isna(merged_df['TARGET'])]
print('merged_df_train.shape is ', merged_df_train.shape,
      'merged_df_test.shape is ', merged_df_test.shape)

train_y = merged_df_train['TARGET']
merged_X_train = merged_df_train.drop(['TARGET'], axis=1)
merged_X_test = merged_df_test.drop(['TARGET'], axis=1)

print('before filter colums merged_X_train.shape', merged_X_train.shape,
      'merged_X_test.shape ', merged_X_test.shape)

#得到top200个feature 其他的feature全部过滤掉
#feas_imp_df_ = pd.read_csv('C:/D_Disk/data_competition/home_credit/outcome/feas_imp_avg.csv')
#useful_cols = list(feas_imp_df_.loc[:200, 'feature'])
#merged_X_train = merged_X_train[useful_cols]
#merged_X_test = merged_X_test[useful_cols]

print('after filter colums merged_X_train.shape', merged_X_train.shape,
      'merged_X_test.shape ', merged_X_test.shape)

X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(merged_X_train, 
            train_y, test_size=0.2, random_state=42)
print('after train_test_split, X_train_new.shape: ', X_train_new.shape,
      'X_val_new.shape: ', X_val_new.shape)

del merged_df
gc.collect()

def test_param(param):
    global merged_X_test
    print('in test_param()')
    
    start_t = time.time()
    
    lgbm = lgb.LGBMClassifier(**param)
    lgbm.fit(X_train_new, y_train_new, 
             eval_set=[(X_val_new, y_val_new)],
             eval_metric='auc',
             early_stopping_rounds=500,
             verbose=400)
    
    best_iteration = lgbm.best_iteration_
    print('best_iteration: {} partial fit cost time: {} '.format(
           lgbm.best_iteration_, time.time()-start_t))
    
    y_predictions_lgb = lgbm.predict_proba(X_val_new)[:,1]
    auc_lgb = round(roc_auc_score(y_val_new, y_predictions_lgb), 5)
    print('partial datat auc_lgb: {}, cost time: {}'.format(auc_lgb, 
          time.time()-start_t))
    
    param_md5_str = convert_2_md5(param)
    store_path = 'C:/D_Disk/data_competition/home_credit/outcome/'
    partial_file_name = ('_'.join(['submission_partial', str(auc_lgb), param_md5_str])
                         + '.csv')
    full_file_name = ('_'.join(['submission_full', str(auc_lgb), param_md5_str])
                       + '.csv')
    
    merged_X_test = merged_X_test.drop(['TARGET'], axis=1, errors='ignore')
    merged_X_test['TARGET'] = lgbm.predict_proba(merged_X_test)[:,1]
    merged_X_test['TARGET'].to_csv(store_path+partial_file_name, header=['TARGET'])
    
    feature_importance_df = pd.DataFrame()
    feature_importance_df["feature"] = X_train_new.columns
    feature_importance_df["importance"] = lgbm.feature_importances_
    feature_importance_df.sort_values(by="importance", ascending=False, inplace=True)
    feature_importance_df.to_csv(store_path+'feas_imp_one_pass.csv', index=False)
    
    del lgbm, y_predictions_lgb
    gc.collect()
    
    start_t = time.time()
    param['n_estimators'] = (best_iteration + 100) #增加100轮训练
    print('start full data training, ', param['n_estimators'])
    lgbm = lgb.LGBMClassifier(**param)
    lgbm.fit(merged_X_train, train_y)
    merged_X_test = merged_X_test.drop(['TARGET'], axis=1, errors='ignore')
    merged_X_test['TARGET'] = lgbm.predict_proba(merged_X_test)[:,1]
    merged_X_test['TARGET'].to_csv(store_path+full_file_name, header=['TARGET'])
    print('full data train cost time: {}'.format(time.time()-start_t))
    
    del lgbm
    gc.collect()
        
    write_to_log('-'*25, ' md5 value: ', param_md5_str, '-'*25)
    write_to_log('param: ', param)
    write_to_log('best_iteration: ', best_iteration)
    write_to_log('valid aug: ', auc_lgb)
    write_to_log('-'*80+'\n')

#raise Exception('find error!')

#-----------------------------------------------------------------------------

#print('start training with v1')
#
#lgbm = lgb.LGBMClassifier(n_estimators=8000, n_jobs=-1, learning_rate=0.01, 
#                          random_state=42, max_depth=15, min_child_samples=700,
#                          num_leaves=51, subsample=0.8, colsample_bytree=0.6,
#                          silent=-1, verbose=-1)
#
#lgbm.fit(X_train_new, y_train_new, eval_set=[(X_train_new, y_train_new), (X_val_new, y_val_new)], 
#         eval_metric= 'auc', verbose=300, early_stopping_rounds=500)
#start_t = time.time()
#y_predictions_lgb = lgbm.predict_proba(X_val_new)[:,1]
#auc_lgb = roc_auc_score(y_val_new, y_predictions_lgb)
#print('partial datat auc_lgb: {}, cost time: {}'.format(auc_lgb, time.time()-start_t))
#
#merged_X_test['TARGET'] = lgbm.predict_proba(merged_X_test)[:,1]
#merged_X_test['TARGET'].to_csv('C:/D_Disk/data_competition/home_credit/all/outcome_new_partial1.csv',
#          header=['TARGET'])
#
#del lgbm
#gc.collect()

#-----------------------------------------------------------------------------

#
#print('start training with v2')
#lgbm = lgb.LGBMClassifier(n_estimators=8000, n_jobs=-1, learning_rate=0.01, 
#                          random_state=42, max_depth=15, min_child_samples=500,
#                          num_leaves=51, subsample=0.8, colsample_bytree=0.6,
#                          reg_alpha=0.1, reg_lambda=0.3, silent=-1, verbose=-1)
#lgbm.fit(X_train_new, y_train_new, eval_set=[(X_train_new, y_train_new), (X_val_new, y_val_new)], 
#         eval_metric= 'auc', verbose=300, early_stopping_rounds=500)
#del lgbm
#gc.collect()

#-----------------------------------------------------------------------------

#param = {'n_estimators':1000, 'n_jobs':-1, 'learning_rate':0.01,
#         'random_state':42, 'max_depth':15, 'min_child_samples':900,
#         'num_leaves':51, 'subsample':0.8, 'colsample_bytree':0.6,
#         'reg_alpha':0.1, 'reg_lambda':0.3, 'silent':-1, 'verbose':-1}


#param = {'n_estimators':8000, 'n_jobs':-1, 'learning_rate':0.01,
#         'random_state':42, 'max_depth':8, 'min_child_samples':1000,
#         'num_leaves':51, 'subsample':0.8, 'colsample_bytree':0.6,
#         'reg_alpha':0.1, 'reg_lambda':0.3, 'silent':-1, 'verbose':-1}
#test_param(param)
#
   
    
param = {'n_estimators':8000, 'n_jobs':-1, 'learning_rate':0.01,
         'random_state':42, 'max_depth':8, 'min_child_samples':900,
         'num_leaves':51, 'subsample':0.7, 'colsample_bytree':0.6,
         'importance_type':'gain', 'silent':-1, 'verbose':-1}
test_param(param)


#param = {'n_estimators':8000, 'n_jobs':-1, 'learning_rate':0.008,
#         'random_state':42, 'max_depth':8, 'min_child_samples':900,
#         'num_leaves':51, 'subsample':0.8, 'colsample_bytree':0.5,
#         'silent':-1, 'verbose':-1}
#test_param(param)

#param = {'n_estimators':10000, 'n_jobs':-1, 'learning_rate':0.01,
#         'subsample':0.8, 'colsample_bytree':0.6, 'silent':-1,
#         'verbose':-1}
#lgbm = lgb.LGBMClassifier(**param)
#lgbm.fit(X_train_new, y_train_new, eval_set=[(X_val_new, y_val_new)], 
#         eval_metric= 'auc', verbose=400, early_stopping_rounds=500)
#
#y_predictions_lgb = lgbm.predict_proba(X_val_new)[:,1]
#auc_lgb = roc_auc_score(y_val_new, y_predictions_lgb)
#print('partial datat auc_lgb: {}, cost time: {}'.format(auc_lgb, time.time()-start_t))
#
##merged_X_test = merged_X_test.drop(['TARGET'], axis=1)
#merged_X_test = merged_X_test.drop(['TARGET'], axis=1, errors='ignore')
#merged_X_test['TARGET'] = lgbm.predict_proba(merged_X_test)[:,1]
#merged_X_test['TARGET'].to_csv('C:/D_Disk/data_competition/home_credit/all/outcome_new_partial2_new.csv',
#          header=['TARGET'])
#
##-----------------------------------------------------------------------------
#
#merged_X_test = merged_X_test.drop(['TARGET'], axis=1, errors='ignore')
#
#start_t = time.time()
#param['n_estimators'] = (lgbm.best_iteration_ + 50) #增加50轮训练
#print('start full data training, ', param['n_estimators'])
#lgbm = lgb.LGBMClassifier(**param)
#lgbm.fit(merged_X_train, train_y)
#y_predictions_lgb = lgbm.predict_proba(merged_X_train)[:,1]
#auc_lgb = roc_auc_score(train_y, y_predictions_lgb)
#print('full data train auc_lgb: {}, cost time: {}'.format(auc_lgb, 
#      time.time()-start_t))
#
#merged_X_test['TARGET'] = lgbm.predict_proba(merged_X_test)[:,1]
#merged_X_test['TARGET'].to_csv('C:/D_Disk/data_competition/home_credit/all/outcome_new_full.csv',
#          header=['TARGET'])

#-----------------------------------------------------------------------------