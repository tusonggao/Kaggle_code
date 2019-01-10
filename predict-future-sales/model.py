# Kaggle problem:
# https://www.kaggle.com/c/competitive-data-science-predict-future-sales

###########################################################################

# Siliar problems:
# https://www.kaggle.com/c/rossmann-store-sales/data

###########################################################################

# Reference codes:
# https://github.com/anhquan0412/Predict_Future_Sales

###########################################################################

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

##########################################################################

# features:
# 1. 价格
# 2. 商品类别
# 3. 商品名字的长度

# 3. 商家的平均销售数量
# 4. 该商品的平均销售数量
# 5. 该商家某个类别的总销量

##########################################################################

random_seed = 42
np.random.seed(random_seed)



# def rmse(y_true, y_predict):
#     return np.sqrt(np.mean((y_true - y_predict) ** 2))

def rmse(y_true, y_pred):
    y_pred = np.where(y_pred>0, y_pred, 0)
    return 'RMSE', np.sqrt(mean_squared_error(y_true, y_pred)), False

def compute_top_multiple(label_y, predict_y, threshold=10, by_percentage=True):
    df = pd.DataFrame()
    df['label_y'] = label_y
    df['predict_y'] = predict_y
    df.sort_values(by=['predict_y'], ascending=False, inplace=True)
    ratio_whole = sum(df['label_y'])/df.shape[0]
    if by_percentage:        
        df_top = df[:int(threshold*0.01*df.shape[0])]
    else:
        df_top = df[:threshold]        
    ratio_top = sum(df_top['label_y'])/df_top.shape[0]
    ratio_mutiple = ratio_top/ratio_whole
    return ratio_mutiple

def compute_bottom_multiple(label_y, predict_y, threshold=10, by_percentage=True):
    df = pd.DataFrame()
    df['label_y'] = label_y
    df['predict_y'] = predict_y
    df.sort_values(by=['predict_y'], ascending=False, inplace=True)
    ratio_whole = sum(df['label_y'])/df.shape[0]
    if by_percentage:
        df_bottom = df[-int(threshold*0.01*df.shape[0]):]
    else:
        df_bottom = df[-threshold:]        
    ratio_bottom = sum(df_bottom['label_y'])/df_bottom.shape[0]
    ratio_mutiple = ratio_bottom/ratio_whole
    return ratio_mutiple


# df_pos = pd.read_csv('./data/hive_sql_pos_instances_data.csv')
# df_neg = pd.read_csv('./data/hive_sql_neg_instances_data_modified.csv')
# df_neg = df_neg.sample(n=600000)
# df_pos['y'] = 1
# df_neg['y'] = 0
# df_merged = pd.concat([df_pos, df_neg])
# print('df_pos shape is ', df_pos.shape)
# print('df_pos head is ', df_pos.head(3))
# print('df_neg is ', df_neg.shape)
# print('df_neg head is ', df_neg.head(3))

print('hello world')
# df_merged = pd.read_csv('./data/hive_sql_merged_instances.csv', sep='\t')
# df_merged.to_csv('./data/hive_sql_merged_instances_comma.csv', index=0)

# df_merged = pd.read_csv('./data/hive_sql_merged_instances_comma.csv')

# df_train = pd.read_csv('./data/sales_train_v2.csv', dtype={'item_id': str, 'shop_id': str})

data_path = 'F:/git_repos/Kaggle_code/predict-future-sales/data/'

start_t = time.time()
df_train_gz = pd.read_csv(os.path.join(data_path, 'sales_train.csv.gz'))
print('df_train_gz cost time', time.time()-start_t)

start_t = time.time()
df_train = pd.read_csv(os.path.join(data_path, 'sales_train_v2.csv'))
print('df_train cost time', time.time()-start_t)

print('df_train_gz equal df_train', df_train_gz==df_train)

df_train.drop(['date', 'date_block_num'], axis=1, inplace=True)

# df_test = pd.read_csv('./data/test.csv', dtype={'item_id': str, 'shop_id': str}, index_col=0)
df_test = pd.read_csv('./data/test.csv')
df_test_ID = df_test['ID']
df_test.drop(['ID'], axis=1, inplace=True)

df_item_price = df_train[['item_id', 'item_price']].groupby('item_id').agg({'item_price': np.average})
df_item_price = df_item_price.reset_index()

print('df_item_price.shape is ', df_item_price.shape,
      'df_item_price.head(10)', df_item_price.head(10))

# groupby('sex').agg({'tip': np.max, 'total_bill': np.sum})

print('before df_test.shape is ', df_test.shape,
      'before df_test.head(10)', df_test.head(10))


df_test = pd.merge(df_test, df_item_price, how='left', on=['item_id'])
print('after df_test.shape is ', df_test.shape,
      'after df_test.head(10)', df_test.head(10))

# df_train = df_train.sample(frac=0.01)
# df_test = df_test.sample(frac=0.1)
df_merged = pd.concat([df_train, df_test])

print('df_train.shape is ', df_train.shape, 'df_test.shape', df_test.shape,
      'df_merged.shape is', df_merged.shape)

# df_items = pd.read_csv('./data/items.csv', dtype={'item_category_id': str})
df_items = pd.read_csv('./data/items.csv')
df_items['item_name_len'] = df_items['item_name'].str.len()
df_items.drop(['item_name'], axis=1, inplace=True)

df_merged = pd.merge(df_merged, df_items, how='left', on=['item_id'])

print('before get_dummies df_merged.shape is ', df_merged.shape)

df_merged = pd.get_dummies(df_merged)

print('after get_dummies df_merged.shape is ', df_merged.shape)

df_train = df_merged[df_merged['item_cnt_day'].notnull()].copy()
df_test = df_merged[df_merged['item_cnt_day'].isnull()].copy()

print('after df_merged.shape is ', df_merged.shape,
      'df_train.shape is ', df_train.shape,
      'df_test.shape is', df_test.shape)

# df_train.is_copy = False
# df_train.loc[:, 'rand_v'] = np.random.rand(df_train.shape[0])
df_train['rand_v'] = np.random.rand(df_train.shape[0])

df_train_train = df_train[df_train['rand_v']<=0.8]
df_y_train_train = df_train_train['item_cnt_day']
df_X_train_train = df_train_train.drop(['item_cnt_day', 'rand_v'], axis=1)

df_train_val = df_train[df_train['rand_v']>0.8]
df_y_train_val = df_train_val['item_cnt_day']
df_X_train_val = df_train_val.drop(['item_cnt_day', 'rand_v'], axis=1)

# df_train_y = df_train['item_cnt_day']
# df_train_X = df_train.drop(['item_cnt_day'], axis=1)

df_test_X = df_test.drop(['item_cnt_day'], axis=1)

# df_item_categories = pd.read_csv('./data/item_categories.csv')

lgbm_param = {'n_estimators':500, 'n_jobs':-1, 'learning_rate':0.1,
              'random_state':42, 'max_depth':9, 'min_child_samples':50,
              'num_leaves':500, 'subsample':0.8, 'colsample_bytree':0.8,
              'silent':-1, 'verbose':-1}
lgbm = lgb.LGBMRegressor(**lgbm_param)
lgbm.fit(df_X_train_train, df_y_train_train, eval_set=[(df_X_train_train, df_y_train_train),
        (df_X_train_val, df_y_train_val)], eval_metric=rmse,
        verbose=100, early_stopping_rounds=1000)

y_predict = lgbm.predict(df_test_X)
y_predict = np.round(y_predict, 5)
y_predict = np.where(y_predict>0, y_predict, 0)
y_predict = np.where(y_predict<20, y_predict, 20)

df_outcome = pd.DataFrame()
df_outcome['ID'] = df_test_ID
df_outcome['item_cnt_month'] = y_predict

df_outcome.to_csv('./outcome/submission1.csv', index=0)

# print('df_item_price.shape is ', df_item_price.shape,
#       'df_item_price.head(10)', df_item_price.head(10))
#
# print('df_train.shape is ', df_train.shape,
#       'df_train.head(10)', df_train.head(10))
#
#
# print('df_test.shape is ', df_test.shape,
#       'df_test.head(10)', df_test.head(10))

# shop_id,item_id

# print('df_items.shape is ', df_items.shape,
#       'df_items.head(10)', df_items.head(10))

# print('df_item_categories.shape is ', df_item_categories.shape,
#       'df_item_categories.head(10)', df_item_categories.head(10))
#
# print('df_item_categories.dtypes is ', df_item_categories.dtypes)

# df_merged['creation_date'] = pd.to_datetime(df_merged['creation_date'], 
#     format='%Y-%m-%d %H:%M:%S')
# df_merged['gap_days'] = (df_merged['creation_date'] - df_merged['creation_date']).dt.days

# print('df_merged head is ', df_merged['gap_days'].head(10))

# split_by_user_id(df_merged)

# print('\n-------------------------------------\n'
#       '     data preprocess finished          \n'
#       '---------------------------------------\n');
#
# # df_merged = pd.get_dummies(df_merged)
# #为了加快训练速度，进行采样
# # df_merged = df_merged.sample(100000)
#
# df_merged = pd.get_dummies(df_merged, columns=['address_code', 'class_code', 'branch_code'])
# print('afte get_dummies, df_merged.shape is ', df_merged.shape)
#
# df_merged_train, df_merged_test = split_by_user_id(df_merged)
# df_merged_train.drop(['buy_user_id', 'creation_date', 'md5_val'], axis=1, inplace=True)
# df_merged_test.drop(['buy_user_id', 'creation_date', 'md5_val'], axis=1, inplace=True)
#
# print('df_merged_train.shape df_merged_test.shape: ', df_merged_train.shape, df_merged_test.shape)
#
# # df_merged_train = pd.get_dummies(df_merged_train)
# # df_merged_test = pd.get_dummies(df_merged_test)
#
# df_train_y = df_merged_train['y']
# df_train_X = df_merged_train.drop(['y'], axis=1)
#
# df_test_y = df_merged_test['y']
# df_test_X = df_merged_test.drop(['y'], axis=1)
#

# print('start training')
# start_t = time.time()

print('program ends')