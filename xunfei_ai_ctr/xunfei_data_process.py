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

count = 0
user_tags_set = set()
user_tags_dict = {}
print('total len is ', len(merged_df))

row_list = []
user_tags_set = set()
user_tags_counter_dict = {}

for index, row in merged_df.iterrows():
    user_tags_str = row['user_tags']
    count += 1
    user_tags_dict = {}
#    user_tags_dict['instance_id'] = str(index)
        
#    if count%10000==0:
#        print('count is ', count)
#        print('user_tags_str is ', user_tags_str)
    
    for s in str(user_tags_str).split(','):
        if len(s)>0 and s!='nan':
            user_tags_dict['user_tags' + s] = 1
            if ('user_tags' + s) in user_tags_counter_dict:
                user_tags_counter_dict['user_tags' + s] += 1
            else:
                user_tags_counter_dict['user_tags' + s] = 1
            user_tags_set.add(s)
        
    if count<=3000:
        row_list.append(user_tags_dict)
        
print('len of user_tags_set is ', len(user_tags_set))
print('row list len is ', len(row_list))

user_tags_df = pd.DataFrame(row_list, dtype=np.int8)
user_tags_df.index = merged_df.index[:len(row_list)]

#user_tags_df_sparse = user_tags_df.to_sparse()
user_tags_df_sparse = pd.SparseDataFrame(row_list, dtype=np.int8)

user_tags_df.to_csv('C:/D_Disk/data_competition/xunfei_ai_ctr/data/user_tags_df.csv')
user_tags_df.iloc[:1000, :].to_csv('C:/D_Disk/data_competition/xunfei_ai_ctr/data/user_tags_partial_df.csv')

print('user_tags_df.shape is ', user_tags_df.shape)
print('to_csv done')

start_t = time.time()
merged_df = merged_df.iloc[:len(user_tags_df)-1, :]
merged_df = merged_df.join(user_tags_df_sparse, how='left')
print('after join merged_df.shape: {} join cost time:{}',
      merged_df.shape, time.time()-start_t)

print('final len of user_tags_set is ', len(user_tags_set))
with open('C:/D_Disk/data_competition/xunfei_ai_ctr/data/user_tags_new.txt', 'w') as file_w:
    for user_tags in user_tags_set:
        if len(user_tags)>0 and user_tags!='nan':
            file_w.write(user_tags+'\n')
            
print('final len of user_tags_set is ', len(user_tags_set))
with open('C:/D_Disk/data_competition/xunfei_ai_ctr/data/user_tags_counter.txt', 'w') as file_w:
    for key, val in user_tags_counter_dict.items():
        file_w.write(str(key) + ' ' + str(val) +'\n')
        
print('write done')
    




