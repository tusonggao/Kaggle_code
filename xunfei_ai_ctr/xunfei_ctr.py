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

#SEED = 911
#SEED = 777
SEED = 1234

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

def managed_change(val_click_prob, ratio=0.01):
    val_click_prob_local = val_click_prob.copy()
    val_click_prob_local.sort()
    idx = int(len(val_click_prob)*ratio)
    threshold_val = val_click_prob_local[-idx]
    
    val_click_prob_new = val_click_prob.copy()
    val_click_prob_new[val_click_prob_new>=threshold_val] = 1.0
    return val_click_prob_new
    

def generate_user_tags_df(merged_df, keep_tags_num=100):
    start_t = time.time()
    print('into generate_user_tags_df')
    
    count = 0
    user_tags_set = set()
    row_list = []
    user_tags_counter_dict = {}

    for index, row in merged_df.iterrows():
        user_tags_str = row['user_tags']
        count += 1
        user_tags_dict = {}
        
        for s in str(user_tags_str).split(','):
            if len(s)>0 and s!='nan':
                user_tags_dict['user_tags' + s] = 1
                if ('user_tags' + s) in user_tags_counter_dict:
                    user_tags_counter_dict['user_tags' + s] += 1
                else:
                    user_tags_counter_dict['user_tags' + s] = 1
                user_tags_set.add(s)
        row_list.append(user_tags_dict)

    top_user_tags = sorted(user_tags_counter_dict.keys(), 
                        key=lambda x: user_tags_counter_dict[x],
                        reverse=True)
    selected_tags = top_user_tags[:keep_tags_num]
#    print('top_user_tags is ', selected_tags)

    for i in range(len(row_list)):
        new_dict = {}
        for k, v in row_list[i].items():
            if k in selected_tags:
                new_dict[k] = v
        row_list[i] = new_dict
    
    user_tags_df = pd.DataFrame(row_list, dtype=np.int8)
    user_tags_df.index = merged_df.index
    user_tags_df_sparse = user_tags_df.to_sparse()
    
    print('getting out generate_user_tags_df, cost time: ',
           time.time()-start_t, 'mem_usage: {}',
           mem_usage(user_tags_df_sparse))
    
    del user_tags_df
    gc.collect()
    
    return user_tags_df_sparse

def process_make_feature(merged_df, num=100):
    print('in process_make_feature')
    start_t = time.time()
    saved_makes = list(merged_df.make.value_counts(dropna=False)[:num].index)
    merged_df['make'] = merged_df['make'].map(lambda x: 
                            x if x in saved_makes else 'na')
#    print(merged_df['make'].value_counts())
    print('cost time ', time.time()-start_t)
    return merged_df

def process_model_feature(merged_df, num=100):
    print('in process_model_feature')
    start_t = time.time()
    merged_df.model.fillna(value='na', inplace=True)
    saved_models = list(merged_df.model.value_counts(dropna=False)[:num].index)
    merged_df['model'] = merged_df['model'].map(lambda x: 
                            x if x in saved_models else 'na')
    print('after model change ', merged_df.model.value_counts(dropna=False))
    print('merged_df.model.isnull().sum() is ', 
          merged_df.model.isnull().sum())
    print('cost time ', time.time()-start_t)
    
    train_df = merged_df[pd.notna(merged_df['click'])]
    train_df = train_df.sample(frac=0.05, random_state=SEED)
    print('in process_model_feature train_df.shape is ', train_df.shape)
    model_agg = train_df.groupby('model').agg({'click': ['mean']})
    model_agg.columns = ['model_click_mean']
    print('in process_model_feature model_agg.shape is ', model_agg.shape,
          model_agg.columns)
    print('model_agg is ', model_agg)

    print('before join merged_df.shape is ', merged_df.shape)
#    print('merged_df.head(5) is ', merged_df.sample(30, random_state=SEED))
    merged_df = merged_df.join(model_agg, how='left', on='model')
    print('after join merged_df.shape is ', merged_df.shape)
    print('merged_df.head(5) is ',
          merged_df[merged_df.model!='na'].sample(10, random_state=SEED))
    
    
    del train_df
    gc.collect()

    return merged_df

def process_osv_feature(merged_df, num=100):
    print('in process_osv_feature')
    start_t = time.time()
    saved_osvs = list(merged_df.osv.value_counts(dropna=False)[:num].index)
    merged_df['osv'] = merged_df['osv'].map(lambda x: 
                            x if x in saved_osvs else 'na')
    print('cost time ', time.time()-start_t)
    return merged_df


def process_slot_id_feature(merged_df, num=100):
    print('in process_slot_id_feature')
    start_t = time.time()
    saved_ids = list(merged_df['inner_slot_id'].value_counts(dropna=False)[:num].index)
    merged_df['inner_slot_id'] = merged_df['inner_slot_id'].map(
                                 lambda x: x if x in saved_ids else 'na')
    print('cost time ', time.time()-start_t)
    return merged_df


def process_industry_feature(merged_df):
    print('in process_industry_feature')
    start_t = time.time()
    merged_df['advert_industry_first'] = merged_df['advert_industry_inner'].map(
                                     lambda x: x.split('_')[0])
    print('cost time ', time.time()-start_t)
    return merged_df


#def process_time_feature(merged_df):
#    def convert_to_hour(time_stamp):
#        timeArray = time.localtime(time_stamp)
#        time_str = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
#        return int(time_str.split()[1].split(':')[0])
#    print('in process_time_feature')
#    start_t = time.time()
#    merged_df['hour'] = merged_df['time'].map(convert_to_hour)
#    
#    merged_df['is_midnight'] = 0
#    merged_df.loc[(merged_df['hour']>=23) | (merged_df['hour']<=4), 'is_midnight'] = 1
#    
#    merged_df['is_afternoon'] = 0
#    merged_df.loc[(merged_df['hour']>=13) & (merged_df['hour']<=17), 'is_afternoon'] = 1
#    
#    merged_df['is_morning'] = 0
#    merged_df.loc[(merged_df['hour']>=7) & (merged_df['hour']<=11), 'is_morning'] = 1
#    
#    print('cost time ', time.time()-start_t)
#    return merged_df

def process_time_feature(merged_df):
    def convert_to_hour(time_stamp):
        timeArray = time.localtime(time_stamp)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        return int(time_str.split()[1].split(':')[0])
    print('in process_time_feature')
    start_t = time.time()
    merged_df['hour'] = merged_df['time'].map(convert_to_hour)
    
    merged_df['is_midnight'] = 0
    merged_df.loc[(merged_df['hour']>=23) | (merged_df['hour']<=4), 'is_midnight'] = 1
    
    merged_df['is_afternoon'] = 0
    merged_df.loc[(merged_df['hour']>=13) & (merged_df['hour']<=17), 'is_afternoon'] = 1
    
    merged_df['is_morning'] = 0
    merged_df.loc[(merged_df['hour']>=7) & (merged_df['hour']<=11), 'is_morning'] = 1
    
    merged_df['hour'] = merged_df['hour'].map(str)
    
    print('cost time ', time.time()-start_t)
    return merged_df

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
    value = str(value) + str(SEED)
    md5_str = hashlib.md5(value.encode('utf-8')).hexdigest()
    return md5_str[:15]

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

del best_dtypes_df, dtypes_dict
gc.collect()

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

#user_tags_df_sparse = generate_user_tags_df(merged_df, keep_tags_num=300)
#user_tags_df_sparse = generate_user_tags_df(merged_df, keep_tags_num=150)

start_t = time.time()
merged_df.sort_values(by=['time'], ascending=False, inplace=True)
merged_df.drop(['user_tags'], axis=1, inplace=True)

merged_df = process_make_feature(merged_df, num=100)
#merged_df = process_model_feature(merged_df, num=220)
merged_df = process_model_feature(merged_df, num=5)
merged_df = process_osv_feature(merged_df, num=80)
merged_df = process_slot_id_feature(merged_df, num=80)
merged_df = process_time_feature(merged_df)
merged_df = process_industry_feature(merged_df)

merged_df = pd.get_dummies(merged_df, 
                columns=['city', 'province', 'os_name', 'f_channel', 
                         'advert_name', 'creative_type', 'advert_id',
                         'app_cate_id', 'advert_industry_first',
                         'advert_industry_inner', 'carrier', 'nnt',
                         'devtype', 'os','make', 'model', 'osv', 
                         'inner_slot_id', 'hour'])
    
print('merged_df.shape 222 is {} merge data cost time:{}, mem_usage:{}'.format(
       merged_df.shape, time.time()-start_t, mem_usage(merged_df)))

#start_t = time.time()
#merged_df = merged_df.join(user_tags_df_sparse, how='left')
#print('after join merged_df.shape: {} join cost time:{} mem_usage:{}'.format(
#      merged_df.shape, time.time()-start_t, mem_usage(merged_df)))
#del user_tags_df_sparse
#gc.collect()

#merged_df.info(memory_usage='deep')


train_df = merged_df[pd.notna(merged_df['click'])]
train_df = train_df.sample(frac=0.05, random_state=SEED)

#print('train_df.shape is ', train_df.shape)
#model_agg = train_df.groupby('model').agg({'click': ['mean']})
#print('model_agg.shape is ', model_agg.shape)

train_y = train_df['click']
train_X = train_df.drop(['click', 'time'], axis=1)

train_df_split_train = train_df[train_df['time']<2190556800]
train_df_split_val = train_df[train_df['time']>=2190556800]

y_train_new = train_df_split_train['click']
X_train_new = train_df_split_train.drop(['click', 'time'], axis=1)
y_val_new = train_df_split_val['click']
X_val_new = train_df_split_val.drop(['click', 'time'], axis=1)
print('X_train_new.shape is {}, X_val_new.shape is {}'.format(
       X_train_new.shape, X_val_new.shape))

del train_df, train_df_split_train, train_df_split_val
gc.collect()

test_df = merged_df[pd.isna(merged_df['click'])]
test_X = test_df.drop(['click', 'time'], axis=1)

outcome_df = pd.DataFrame()
outcome_df['instance_id'] = test_df.index
outcome_df.set_index('instance_id', inplace=True)

del merged_df, test_df
gc.collect()

def generate_learning_rate_list(num=2000):
    start_rate = 0.101
    lst = [start_rate]*150
    for i in range(1, 50):
        current_rate = start_rate - i*0.001
        if current_rate<0.04:
            break
        lst += [current_rate]*25
    if num>len(lst):
        lst += [0.04]*(num-len(lst))
#    return lst[:num]
    return [0.09]*num

def test_param(lgbm_param):
    print('in test_param')
    gc.collect()
    global y_train_new, X_train_new, y_val_new, X_val_new
    
#    val_num = int(0.2*len(train_X))
#    X_train_new = train_X.iloc[val_num:,:]
#    y_train_new = train_y[val_num:]
#    X_val_new = train_X.iloc[:val_num,:]
#    y_val_new = train_y.iloc[:val_num]
    
#    print('start train_test_split')
#    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_X, 
#                train_y, test_size=0.25, random_state=SEED)
    
    start_t = time.time()
    lgbm = lgb.LGBMClassifier(**lgbm_param)
    learning_rate_func = lgb.reset_parameter(
                       learning_rate = generate_learning_rate_list())
    print('start partial trainning')
    lgbm.fit(X_train_new, y_train_new, 
             eval_set=[(X_train_new, y_train_new), (X_val_new, y_val_new)],
             callbacks=[learning_rate_func],
#            eval_metric=log_loss_def, 
             eval_metric='logloss',
#            eval_metric='auc', 
             verbose=100, 
             early_stopping_rounds=300)
    print('partial fit cost time: ', time.time()-start_t)
    
    best_iteration = lgbm.best_iteration_
#    print('best score value is ', lgbm.best_score_)
#    logloss_val = round(lgbm.best_score_['valid_1']['auc'], 5)
    logloss_val = round(lgbm.best_score_['valid_1']['binary_logloss'], 5)
    
    val_click_prob = lgbm.predict_proba(X_val_new)[:,1]
#    val_click_prob_new = managed_change(val_click_prob)
    print('after managed_change logloss is ',
          log_loss(y_val_new, val_click_prob),
          log_loss(y_val_new, managed_change(val_click_prob, ratio=0.01)),
          log_loss(y_val_new, managed_change(val_click_prob, ratio=0.02)),
          log_loss(y_val_new, managed_change(val_click_prob, ratio=0.05)),
          log_loss(y_val_new, managed_change(val_click_prob, ratio=0.08)),
          log_loss(y_val_new, managed_change(val_click_prob, ratio=0.10))
          )
    
    print('after managed_change auc is ',
          roc_auc_score(y_val_new, val_click_prob),
          roc_auc_score(y_val_new, managed_change(val_click_prob, ratio=0.01)),
          roc_auc_score(y_val_new, managed_change(val_click_prob, ratio=0.02)),
          roc_auc_score(y_val_new, managed_change(val_click_prob, ratio=0.05)),
          roc_auc_score(y_val_new, managed_change(val_click_prob, ratio=0.08)),
          roc_auc_score(y_val_new, managed_change(val_click_prob, ratio=0.10))
          )
       
    start_t = time.time()
    prediction_click_prob = lgbm.predict_proba(test_X)[:,1]
    outcome_df['predicted_score'] = prediction_click_prob
    
    
    param_md5_str = convert_2_md5(lgbm_param)
    store_path = 'C:/D_Disk/data_competition/xunfei_ai_ctr/outcome/'
    partial_file_name = '_'.join(['submission_partial', str(logloss_val), param_md5_str]) + '.csv'
    full_file_name = '_'.join(['submission_full', str(logloss_val), param_md5_str]) + '.csv'
    
    outcome_df['predicted_score'].to_csv(store_path+partial_file_name,
           header=['predicted_score'])
    print('partial get predict outcome cost time: ', time.time()-start_t)
    
    del lgbm
    gc.collect()
    del X_train_new, X_val_new
    gc.collect()
    del y_train_new, y_val_new
    gc.collect()
    for i in range(5):
        gc.collect()
    
#    start_t = time.time()
#    lgbm_param['n_estimators'] = int(best_iteration*1.0)
#    print('normal full fit n_estimators is ', int(best_iteration*1.0))
#    lgbm = lgb.LGBMClassifier(**lgbm_param)
#    lgbm.fit(train_X, train_y)
#    print('normal full fit cost time: ', time.time()-start_t)
#    
#    start_t = time.time()
#    prediction_click_prob = lgbm.predict_proba(test_X)[:,1]
#    outcome_df['predicted_score'] = prediction_click_prob
#    outcome_df['predicted_score'].to_csv(store_path+full_file_name,
#           header=['predicted_score'])
#    print('normal full predict cost time: ', time.time()-start_t)
    
    start_t = time.time()
    lgbm_param['n_estimators'] = int(best_iteration*1.1)
    print('extra full fit n_estimators is ', int(best_iteration*1.1))
    lgbm = lgb.LGBMClassifier(**lgbm_param)
    
    learning_rate_func = lgb.reset_parameter(learning_rate = 
        generate_learning_rate_list()[:lgbm_param['n_estimators']])
    
    lgbm.fit(train_X, train_y,
             callbacks=[learning_rate_func])
    print('extra full fit cost time: ', time.time()-start_t)
    
    start_t = time.time()
    prediction_click_prob = lgbm.predict_proba(test_X)[:,1]
    outcome_df['predicted_score'] = prediction_click_prob
    outcome_df['predicted_score'].to_csv(store_path + full_file_name,
           header=['predicted_score'])
    print('extra full predict cost time: ', time.time()-start_t)
    
    write_to_log('-'*25, ' md5 value: ', param_md5_str, '-'*25)
    write_to_log('param: ', lgbm_param)
    write_to_log('best_iteration: ', best_iteration)
    write_to_log('valid rmse: ', logloss_val)
    write_to_log('-'*80+'\n')
    
def test_param_xgboost():
    start_t = time.time()
    xgbm = xgb.XGBClassifier(max_depth=5, n_estimators=200, n_jobs=3, 
                             learning_rate=0.08, random_state=42,
                             silent=False, subsample=0.6, 
                             colsample_bytree=0.7)
    print('start partial trainning')
    xgbm.fit(X_train_new, y_train_new, eval_set=[(X_train_new, y_train_new), 
            (X_val_new, y_val_new)], eval_metric='logloss',
            verbose=5, early_stopping_rounds=60)
    print('partial fit cost time: ', time.time()-start_t)
    
    evals_result = xgbm.evals_result()
    print('xgbm evals_result is ', evals_result)
    
    val_click_prob = xgbm.predict_proba(X_val_new)[:,1]
    print('after managed_change logloss is ',
          log_loss(y_val_new, val_click_prob),
          log_loss(y_val_new, managed_change(val_click_prob, ratio=0.01)),
          log_loss(y_val_new, managed_change(val_click_prob, ratio=0.02)),
          log_loss(y_val_new, managed_change(val_click_prob, ratio=0.05)),
          log_loss(y_val_new, managed_change(val_click_prob, ratio=0.08)),
          log_loss(y_val_new, managed_change(val_click_prob, ratio=0.10))
          )
    
#    best_iteration = xgbm.best_iteration_
    
    print('best score value is ', xgbm.best_score_)
    print('partial fit cost time ', time.time()-start_t)
#    logloss_val = round(xgbm.best_score_['valid_1']['LOG_LOSS'], 5)
    
#    xgbm.fit(train_X, train_y)
#    y_predprob_xgb = xgbm.predict_proba(test_X)[:,1]
#    auc_xgboost = roc_auc_score(test_y, y_predprob_xgb)
#    print('xgboost acc_rate is ', acc_rate_xgboost, auc_xgboost)
#    end_t_xgb = time.time()
#    print('xgb total cost time is ', end_t_xgb - start_t_xgb, ' seconds')
#    feature_matrix = xgbm.feature_importances_

#lgbm_param = {'n_estimators':800, 'n_jobs':-1, 'learning_rate':0.08,
#              'random_state':SEED, 'max_depth':6, 'min_child_samples':71,
#              'num_leaves':31, 'subsample':0.75, 'colsample_bytree':0.8,
#              'subsample_freq':1, 'silent':-1, 'verbose':-1}

#lgbm_param = {'n_estimators':2000, 'n_jobs':-1, 'learning_rate':0.09,
#              'random_state':SEED, 'max_depth':6, 'min_child_samples':71,
#              'num_leaves':31, 'subsample':0.75, 'colsample_bytree':0.2,
#              'subsample_freq':3, 'silent':-1, 'verbose':-1}

lgbm_param = {'n_estimators':2000, 'n_jobs':-1, 'learning_rate':0.09,
              'random_state':SEED, 'max_depth':6, 'min_child_samples':71,
              'num_leaves':31, 'subsample':0.75, 'colsample_bytree':0.2,
              'subsample_freq':3, 'silent':-1, 'verbose':-1}

#lgbm_param = {'n_estimators':2000, 'n_jobs':-1, 'learning_rate':0.09,
#              'random_state':SEED, 'max_depth':5, 'min_child_samples':71,
#              'num_leaves':31, 'subsample':0.75, 'colsample_bytree':0.1,
#              'subsample_freq':4, 'silent':-1, 'verbose':-1}

#test_param(lgbm_param)

#test_param_xgboost()

