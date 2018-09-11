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
from sklearn.metrics import log_loss

from sklearn.model_selection import train_test_split

from pandas.api.types import is_numeric_dtype

import warnings
warnings.simplefilter(action='ignore')

SEED = 911

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def generate_user_tags_df(merged_df):
    start_t = time.time()
    print('into generate_user_tags_df')
    row_list = []
    count = 0
    for index, row in merged_df.iterrows():
        user_tags_str = row['user_tags']
        count += 1
        user_tags_dict = {}
        for s in str(user_tags_str).split(','):
            if len(s)>0 and s!='nan':
                user_tags_dict['user_tags' + s] = 1
        row_list.append(user_tags_dict)

    print('row list len is ', len(row_list))
    user_tags_df = pd.DataFrame(row_list, dtype=np.int8)
    user_tags_df.index = merged_df.index
    print('geting out of generate_user_tags_df, cost time: ',
          time.time()-start_t, 'user_tags_df.shape is ',
          user_tags_df.shape, 'mem_usage is ',
          mem_usage(user_tags_df))
    
    user_tags_df_sparse = user_tags_df.to_sparse()
    
    del user_tags_df
    gc.collect()
    
    return user_tags_df_sparse


def generate_numeric_dtypes(df, store_file_path=None):
    int_types = ["uint8", "int8", "int16", 'int32', 'int64']
    int_types_min_max_list = []
    for it in int_types:
        int_types_min_max_list.append((it, np.iinfo(it).min, np.iinfo(it).max))
    
    float_types = ['float16', 'float32', 'float64']
    float_types_min_max_list = []
    for ft in float_types:
        float_types_min_max_list.append((ft, np.finfo(ft).min, np.finfo(ft).max))
    
    dtypes_dict = dict(df.dtypes)
    dtypes_dict_new = dict()
    
    for col_name, d_t in dtypes_dict.items():
        if is_numeric_dtype(df[col_name])==False:
            continue
        if d_t==np.float64 or d_t==np.float32:
            for ft, min_v, max_v in float_types_min_max_list:
                if min_v <= df[col_name].min() <= df[col_name].max() <= max_v:
                    dtypes_dict_new[col_name] = ft
                    break
        elif d_t==np.int64 or d_t==np.int32 or d_t==np.int16:
            for it, min_v, max_v in int_types_min_max_list:
                if min_v <= df[col_name].min() <= df[col_name].max() <= max_v:
                    dtypes_dict_new[col_name] = it
                    break
        else:
            dtypes_dict_new[col_name] = d_t
    
    dtypes_df = pd.DataFrame.from_dict(
                     {'col_name': list(dtypes_dict_new.keys()),
                      'best_dtype': list(dtypes_dict_new.values())}
                )
    dtypes_df = dtypes_df.set_index('col_name')
    print('dtypes_df.shape: ', dtypes_df.shape)
    if store_file_path is not None:
        dtypes_df.to_csv(store_file_path)
    return dtypes_dict_new

def convert_2_md5(value):
    return hashlib.md5(str(value).encode('utf-8')).hexdigest()

def write_to_log(*param):
    param_list = [str(s) for s in param]
    log = ' '.join(param_list)
    with open('./outcome/log_file.txt', 'a') as file:
        file.write(log+'\n')
        file.flush()  #立即写入磁盘
        os.fsync(file)  #立即写入磁盘

def log_loss_def(y_true, y_pred):        
    return 'LOG_LOSS', log_loss(y_true, y_pred), False

def log_loss_tsg(true_y, y_pred):
    return -np.mean(true_y*np.log(y_pred)+ (1-true_y)*np.log(1-y_pred))

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

best_dtypes_df = pd.read_csv(
    'C:/D_Disk/data_competition/xunfei_ai_ctr/data/merged_df_dtypes.csv',
    index_col=0)
dtypes_dict = dict(best_dtypes_df['best_dtype'])

#merged_df = pd.read_csv('C:/D_Disk/data_competition/gamer_value/data/merged_df.csv', 
#                       index_col=0, header=0, dtype=dtypes_dict)

train_df = pd.read_csv('C:/D_Disk/data_competition/xunfei_ai_ctr/data/round1_iflyad_train.txt', 
                        sep='\t', index_col=0, header=0, dtype=dtypes_dict,
                        engine='python', encoding='utf-8')
test_df = pd.read_csv('C:/D_Disk/data_competition/xunfei_ai_ctr/data/round1_iflyad_test_feature.txt', 
                       sep='\t', index_col=0, header=0, dtype=dtypes_dict,
                       engine='python', encoding='utf-8')

print('train_df.shape is {} test_df.shape is {} load data cost time:{}'.format(
       train_df.shape, test_df.shape, time.time()-start_t))

start_t = time.time()
merged_df = train_df.append(test_df)

print('merged_df.shape 111 is {} merge data cost time:{}'.format(
       merged_df.shape, time.time()-start_t))

del train_df, test_df
gc.collect()

#merged_df = pd.read_csv('C:/D_Disk/data_competition/gamer_value/data/merged_df.csv', 
#                       index_col=0, header=0)


#merged_df.info(memory_usage='deep')

#generate_numeric_dtypes(
#    merged_df,
#    'C:/D_Disk/data_competition/xunfei_ai_ctr/data/merged_df_dtypes.csv'
#)

user_tags_df = generate_user_tags_df(merged_df)

start_t = time.time()
merged_df.drop(['advert_industry_inner', 'osv', 'make', 'user_tags', 
                'model', 'inner_slot_id'],
                 axis=1, inplace=True)
merged_df = pd.get_dummies(merged_df)
print('merged_df.shape 222 is {} merge data cost time:{}'.format(
       merged_df.shape, time.time()-start_t))

#user_tags_list = []
#with open('C:/D_Disk/data_competition/xunfei_ai_ctr/data/user_tags.txt') as file:
#    for line in file:
#        user_tags_list.append(line.strip())
#user_tags_dtype_dict = {('user_tags'+s):np.int8 for s in user_tags_list}
#print('user_tags_dtype_dict is ', user_tags_dtype_dict)

start_t = time.time()
merged_df = merged_df.join(user_tags_df, how='left')
print('after join merged_df.shape: {} join cost time:{} mem_usage:{}',
      merged_df.shape, time.time()-start_t, mem_usage(merged_df))
del user_tags_df
gc.collect()

#merged_df.info(memory_usage='deep')

#train_df = merged_df[merged_df['prediction_pay_price']!=-99999]
#train_y = train_df['prediction_pay_price'].values
#train_X = train_df.drop(['prediction_pay_price'], axis=1).values

train_df = merged_df[pd.notna(merged_df['click'])]
train_y = train_df['click']
train_X = train_df.drop(['click'], axis=1)
del train_df
gc.collect()

test_df = merged_df[pd.isna(merged_df['click'])]
test_X = test_df.drop(['click'], axis=1)

#test_X = test_df.drop(['click'], axis=1).values
outcome_df = pd.DataFrame()
outcome_df['instance_id'] = test_df.index
outcome_df.set_index('instance_id', inplace=True)

del merged_df, test_df
gc.collect()

X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_X, 
            train_y, test_size=0.25, random_state=SEED)
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
    print('in test_param')
    gc.collect()
    
    lgbm = lgb.LGBMClassifier(**lgbm_param)
    lgbm.fit(X_train_new, y_train_new, eval_set=[(X_train_new, y_train_new), 
            (X_val_new, y_val_new)], eval_metric=log_loss_def, 
            verbose=100, early_stopping_rounds=300)
    
    gc.collect()
    best_iteration = lgbm.best_iteration_
    y_predictions_whole = lgbm.predict_proba(train_X)[:,1]
    RMSLE_score_lgb_whole = round(log_loss(train_y, y_predictions_whole), 5)
    logloss_score_tsg = round(log_loss_tsg(train_y, y_predictions_whole), 5)
    
    gc.collect()
    
    y_predictions_train = lgbm.predict_proba(X_train_new)[:,1]
    RMSLE_score_lgb_train = round(log_loss(y_train_new, y_predictions_train), 5)
    
    y_predictions_val = lgbm.predict_proba(X_val_new)[:,1]
    RMSLE_score_lgb_val = round(log_loss(y_val_new, y_predictions_val), 5)
    
    len_to_get = int(0.20*len(y_val_new))
    RMSLE_score_lgb_val_20_percent = log_loss(y_val_new[:len_to_get], y_predictions_val[:len_to_get])
    
    print('partial data whole_score: {} logloss_score_tsg: {} '
          'train score: {}  test score: {}, '
          'RMSLE_score_lgb_val_20_percent: {}'.format(
           RMSLE_score_lgb_whole, logloss_score_tsg, 
           RMSLE_score_lgb_train, RMSLE_score_lgb_val,
           RMSLE_score_lgb_val_20_percent))
    
    start_t = time.time()
    prediction_click_prob = lgbm.predict_proba(test_X)[:,1]
    outcome_df['predicted_score'] = prediction_click_prob
    
    lgbm_param['n_estimators'] = int(best_iteration*1.1)
    print('full fit n_estimators is ', int(best_iteration*1.1))
    param_md5_str = convert_2_md5(lgbm_param)
    store_path = 'C:/D_Disk/data_competition/xunfei_ai_ctr/outcome/'
    partial_file_name = '_'.join(['submission_partial', str(RMSLE_score_lgb_val), param_md5_str]) + '.csv'
    full_file_name = '_'.join(['submission_full', str(RMSLE_score_lgb_val), param_md5_str]) + '.csv'
    
    outcome_df['predicted_score'].to_csv(store_path+partial_file_name,
           header=['predicted_score'])
    print('partial get predict outcome cost time: ', time.time()-start_t)
    
    gc.collect()
    start_t = time.time()
    lgbm = lgb.LGBMClassifier(**lgbm_param)
    lgbm.fit(train_X, train_y)
    print('full fit cost time: ', time.time()-start_t)
    
    start_t = time.time()
    prediction_click_prob = lgbm.predict_proba(test_X)[:,1]
    outcome_df['predicted_score'] = prediction_click_prob
    outcome_df['predicted_score'].to_csv(store_path+full_file_name,
           header=['predicted_score'])
    
    print('full predict cost time: ', time.time()-start_t)
    
    write_to_log('-'*25, ' md5 value: ', param_md5_str, '-'*25)
    write_to_log('param: ', lgbm_param)
    write_to_log('best_iteration: ', best_iteration)
    write_to_log('valid rmse: ', RMSLE_score_lgb_val)
    write_to_log('-'*80+'\n')

lgbm_param = {'n_estimators':600, 'n_jobs':-1, 'learning_rate':0.08, 
              'random_state':SEED, 'max_depth':6, 'min_child_samples':1001,
              'num_leaves':31, 'subsample':0.75, 'colsample_bytree':0.8,
              'subsample_freq':1, 'silent':-1, 'verbose':-1}

test_param(lgbm_param)

