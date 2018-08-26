import pickle
import time
import gc
import hashlib
import os

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

start_t = time.time()
data_dir_path = 'C:/D_Disk/data_competition/home_credit/data/'
merged_df = pd.read_csv(data_dir_path + '/processed/merged_df.csv',
                        index_col=0)
print('previous merged_df.shape is ', merged_df.shape,
      'read cost time: ', time.time()-start_t)

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

def display_importances(feature_importance_df_):
    outcome_path = 'C:/D_Disk/data_competition/home_credit/outcome/'
    feature_importance_df_.to_csv(outcome_path + 'feas_imp_folds.csv', index=False)
    
    feas_imp_avg_df = (feature_importance_df_[["feature", "importance"]].
                       groupby("feature").mean().
                       sort_values(by="importance", ascending=False))
    feas_imp_avg_df.to_csv(outcome_path+'feas_imp_avg.csv')
    
    cols = (feas_imp_avg_df[:40].index)
    best_features = (feature_importance_df_.
                     loc[feature_importance_df_.feature.isin(cols)])
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout
    plt.savefig('lgbm_importances01.png')

def test_param_folds(param, folds=5):
    global merged_X_test
    print('in test_param()')
    
    start_t = time.time()
    
    feature_importance_df = pd.DataFrame()
    
    auc_score_list = []
    y_pred_result = 0
    for fold in range(folds):
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
                merged_X_train, train_y, test_size=0.2, random_state=fold)
        param['random_state'] = fold
        lgbm = lgb.LGBMClassifier(**param)
        lgbm.fit(X_train_new, y_train_new, eval_set=[(X_val_new, y_val_new)], 
                eval_metric='auc', verbose=200, early_stopping_rounds=500)
        
#        lgbm.fit(X_train_new, y_train_new, eval_set=[(X_val_new, y_val_new)], 
#                eval_metric='auc', verbose=400, early_stopping_rounds=500)
        
        y_predictions_val = lgbm.predict_proba(X_val_new)[:,1]
        auc_score = round(roc_auc_score(y_val_new, y_predictions_val), 5)
        auc_score_list.append(auc_score)
        print('round: ', fold, ' auc for validation data is ', auc_score)
        
        merged_X_test = merged_X_test.drop(['TARGET'], axis=1, errors='ignore')
        y_pred = lgbm.predict_proba(merged_X_test)[:,1]
        y_pred_result += y_pred
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = X_train_new.columns
        fold_importance_df["importance"] = lgbm.feature_importances_
        fold_importance_df["fold"] = fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        del lgbm, X_train_new, X_val_new, y_train_new, y_val_new
        gc.collect()
        
    auc_score_avg = round(np.array(auc_score_list).mean(), 5)
    print('auc_score_list is ', auc_score_list, 
          'auc_score_avg is ', auc_score_avg)
    
    display_importances(feature_importance_df)
    
    y_pred_result /= folds
    merged_X_test['TARGET'] = y_pred_result
    
    param_md5_str = convert_2_md5(param)
    store_path = 'C:/D_Disk/data_competition/home_credit/outcome/'
    folds_file_name = ('_'.join(['submission_folds', str(auc_score_avg), param_md5_str])
                        + '.csv')
    merged_X_test['TARGET'].to_csv(store_path+folds_file_name, header=['TARGET'])
    

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
#         'random_state':42, 'max_depth':8, 'min_child_samples':900,
#         'num_leaves':51, 'subsample':0.8, 'colsample_bytree':0.6,
#         'reg_alpha':0.1, 'reg_lambda':0.3, 'silent':-1, 'verbose':-1}
#test_param_folds(param)

param = {'n_estimators':500, 'n_jobs':-1, 'learning_rate':0.01,
         'random_state':42, 'max_depth':8, 'min_child_samples':900,
         'num_leaves':51, 'subsample':0.8, 'colsample_bytree':0.6,
         'importance_type':'gain', 'silent':-1, 'verbose':-1}
test_param_folds(param)

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