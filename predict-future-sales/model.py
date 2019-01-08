# https://www.kaggle.com/c/competitive-data-science-predict-future-sales

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

def convert_2_md5(value):
    return hashlib.md5(str(value).encode('utf-8')).hexdigest()

def split_by_user_id(df_merged, train_ratio=0.67):
    df_merged['md5_val'] = df_merged['buy_user_id'].apply(convert_2_md5)

    print('df_merged.dtypes is ', df_merged.dtypes)

    df_merged_sorted = df_merged.sort_values(by=['md5_val'])
    # print('df_merged_sorted head is ', df_merged_sorted.head(5))
    # df_merged_sorted.to_csv('./data/hive_sql_merged_instances_sorted.csv', sep='\t', date_format='%Y/%m/%d', index=0)  # date_format='%Y-%m-%d %H:%M:%s'
    row_n = df_merged.shape[0]
    train_num = int(row_n*train_ratio)
    pivot_val = df_merged_sorted.ix[train_num, 'md5_val']
    pivot_val = 'ac3a1976ceca523950645655fd18a927'
    # print('train_num is: ', train_num, 'pivot_val is: ', pivot_val)

    df_merged_train = df_merged_sorted[df_merged_sorted['md5_val']<=pivot_val]
    df_merged_test = df_merged_sorted[df_merged_sorted['md5_val']>pivot_val]
    # df_merged_train.to_csv('./data/hive_sql_merged_instances_train.csv', sep='\t', index=0)
    # df_merged_test.to_csv('./data/hive_sql_merged_instances_test.csv', sep='\t', index=0)

    return df_merged_train, df_merged_test


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
df_train = pd.read_csv('./data/sales_train_v2.csv')
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

df_train = df_merged[df_merged['item_cnt_day'].notnull()]
df_test = df_merged[df_merged['item_cnt_day'].isnull()]

print('after df_merged.shape is ', df_merged.shape,
      'df_train.shape is ', df_train.shape,
      'df_test.shape is', df_test.shape)

df_train.loc[:, 'rand_v'] = np.random.rand(df_train.shape[0])

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

lgbm_param = {'n_estimators':500, 'n_jobs':-1, 'learning_rate':0.08,
              'random_state':42, 'max_depth':7, 'min_child_samples':21,
              'num_leaves':300, 'subsample':0.8, 'colsample_bytree':0.8,
              'silent':-1, 'verbose':-1}
lgbm = lgb.LGBMRegressor(**lgbm_param)
lgbm.fit(df_X_train_train, df_y_train_train, eval_set=[(df_X_train_train, df_y_train_train),
        (df_X_train_val, df_y_train_val)], eval_metric=rmse,
        verbose=100, early_stopping_rounds=1000)

y_predict = lgbm.predict(df_test_X)

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

##----------------------------------###
##------ 加上category_id的feature  -----###
##----------------------------###

# df_train = pd.merge(df_merged, df_items, how='left', on=['item_id'])
# df_merged.drop(['recency_date'], axis=1, inplace=True)
# print('df_merged.shape after add R is ', df_merged.shape)
# print('df_merged.dtypes after add R is ', df_merged.dtypes)

# df_merged.drop(['gap_days'], axis=1, inplace=True)
#
# ###----------------------------###
# ###------ 加上F的feature  -----###
# ###----------------------------###
# df_frequency = pd.read_csv('./data/hive_sql_F_data.csv', parse_dates=[1], infer_datetime_format=True)
# df_merged = pd.merge(df_merged, df_frequency, how='left', on=['buy_user_id', 'creation_date'])
# print('df_merged.shape after add frequency is ', df_merged.shape)
# print('df_merged.dtypes after add frequency is ', df_merged.dtypes)
#
# ###----------------------------###
# ###------ 加上M的feature  -----###
# ###----------------------------###
# df_monetary = pd.read_csv('./data/hive_sql_M_data.csv', parse_dates=[1], infer_datetime_format=True)
# df_merged = pd.merge(df_merged, df_monetary, how='left', on=['buy_user_id', 'creation_date'])
# print('df_merged.shape after add monetary is ', df_merged.shape)
# print('df_merged.dtypes after add monetary is ', df_merged.dtypes)
#
#
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
#
# print('start training')
# start_t = time.time()
#
# lgbm = lgb.LGBMClassifier(n_estimators=500, n_jobs=-1, learning_rate=0.08,
#                          random_state=42, max_depth=13, min_child_samples=400,
#                          num_leaves=100, subsample=0.7, colsample_bytree=0.85,
#                          silent=-1, verbose=-1)
#
# # lgbm.fit(X_train_new, y_train_new, eval_set=[(X_val_new, y_val_new)],
# #         eval_metric='auc', verbose=200, early_stopping_rounds=600)
#
# lgbm.fit(df_train_X, df_train_y)
# print('training ends, cost time: ', time.time()-start_t)
#
# start_t = time.time()
# print('predict starting')
# y_predictions = lgbm.predict_proba(df_test_X)
# auc_score = roc_auc_score(df_test_y, y_predictions[:, 1])
#
#
#
# print('auc_score is ', auc_score, 'predict cost time:', time.time()-start_t)
# print('top 200 ratio_multiple is',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=200, by_percentage=False),
#       'top 500 ratio_multiple is',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=500, by_percentage=False),
#       'ratio_multiple top 1 is ',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=1),
#       'ratio_multiple top 5 is ',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=5),
#       'ratio_multiple top 10 is ',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=10),
#       'ratio_multiple top 20 is ',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=20),
#       'ratio_multiple top 30 is',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=30),
#       'ratio_multiple top 40 is',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=40),
#       'ratio_multiple top 50 is',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=50),
#       'bottom 200 ratio_multiple is',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=200, by_percentage=False),
#       'ratio_multiple bottom 1 is ',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=1),
#       'ratio_multiple bottom 5 is ',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=5),
#       'ratio_multiple bottom 10 is ',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=10),
#       'ratio_multiple bottom 20 is ',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=20),
#       'ratio_multiple bottom 30 is',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=30),
#       'ratio_multiple bottom 40 is',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=40),
#       'ratio_multiple bottom 50 is',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=50),
#       )
#
# # y_pred_proba = clf.predict_proba(df_test)
# # auc = roc_auc_score(y_test, y_pred_proba[:, 1])
#
#
# feature_names = df_train_X.columns.values.tolist()
# df_feat_importance = pd.DataFrame({
#         'column': feature_names,
#         'importance': lgbm.feature_importances_,
#     }).sort_values(by='importance', ascending=False)
# df_feat_importance.to_csv('./model_output/df_feat_importance.csv', index=0, sep='\t')
#
# # df_feat_importance[:22].plot.bar(x='column', y='importance', rot=0)
# # plt.show()
#
# def show_features_importance_bar(features, feature_importance):
#     plt.figure(figsize=(25, 6))
#     #plt.yscale('log', nonposy='clip')
#     plt.bar(range(len(feature_importance)), feature_importance, align='center')
#     plt.xticks(range(len(feature_importance)), features, rotation='vertical')
#     plt.title('Feature importance')
#     plt.ylabel('Importance')
#     plt.xlabel('Features')
#     plt.tight_layout()
#     plt.show()
#
#
# show_features_importance_bar(df_feat_importance['column'][:25],
#                              df_feat_importance['importance'][:25])
#
#
# lgbm = lgb.LGBMClassifier(n_estimators=1000, n_jobs=-1, learning_rate=0.08,
#                          random_state=42, max_depth=7, min_child_samples=500,
#                          num_leaves=55, subsample=0.7, colsample_bytree=0.85,
#                          silent=-1, verbose=-1)
#
# # lgbm.fit(X_train_new, y_train_new, eval_set=[(X_val_new, y_val_new)],
# #         eval_metric='auc', verbose=200, early_stopping_rounds=600)
#
# lgbm.fit(df_train_X, df_train_y)
# print('new training ends, cost time: ', time.time()-start_t)
# start_t = time.time()
# y_predictions = lgbm.predict_proba(df_test_X)
# auc_score = roc_auc_score(df_test_y, y_predictions[:, 1])
#
# print('new auc_score is ', auc_score, 'predict cost time:', time.time()-start_t)
# print('top 200 ratio_multiple is',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=200, by_percentage=False),
#       'top 500 ratio_multiple is',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=500, by_percentage=False),
#       'ratio_multiple top 1 is ',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=1),
#       'ratio_multiple top 5 is ',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=5),
#       'ratio_multiple top 10 is ',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=10),
#       'ratio_multiple top 20 is ',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=20),
#       'ratio_multiple top 30 is',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=30),
#       'ratio_multiple top 40 is',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=40),
#       'ratio_multiple top 50 is',
#       compute_top_multiple(df_test_y, y_predictions[:, 1], threshold=50),
#       'bottom 200 ratio_multiple is',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=200, by_percentage=False),
#       'ratio_multiple bottom 1 is ',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=1),
#       'ratio_multiple bottom 5 is ',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=5),
#       'ratio_multiple bottom 10 is ',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=10),
#       'ratio_multiple bottom 20 is ',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=20),
#       'ratio_multiple bottom 30 is',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=30),
#       'ratio_multiple bottom 40 is',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=40),
#       'ratio_multiple bottom 50 is',
#       compute_bottom_multiple(df_test_y, y_predictions[:, 1], threshold=50),
#       )
#
# print('program ends')