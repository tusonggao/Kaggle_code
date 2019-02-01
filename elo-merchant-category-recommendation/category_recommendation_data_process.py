import pickle
import time
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def count_seconds(t_delta):
    return (60*60*24)*t_delta.days + t_delta.seconds

data_path = './data/elo-merchant-category-recommendation/'

start_t = time.time()
train_df = pd.read_csv(data_path+'train.csv', index_col=0, header=0)
test_df = pd.read_csv(data_path+'test.csv', index_col=0, header=0)
print('train_df.shape is {}, test_df.shape is {} load data cost time:{}'.format(
        train_df.shape, test_df.shape, time.time()-start_t))

test_df['prediction_pay_price'] = -99999
merged_df = train_df.append(test_df)
print('merged_df.shape is ', merged_df.shape)

exit(1)

#merged_df = merged_df.iloc[:1000]

merged_df['register_time'] = pd.to_datetime(merged_df['register_time'], format='%Y-%m-%d %H:%M:%S')
merged_df['after_time'] = merged_df['register_time'] - merged_df['register_time'].min()
merged_df['after_time_seconds'] = merged_df['after_time'].apply(lambda x: count_seconds(x))
merged_df['register_time_month'] = merged_df['register_time'].apply(lambda x: x.month)
merged_df['register_time_day'] = merged_df['register_time'].apply(lambda x: x.day)
merged_df['register_time_hour'] = merged_df['register_time'].apply(lambda x: x.hour)
merged_df.drop(['register_time', 'after_time'], axis=1, inplace=True) # 暂时去掉注册时间
print('with time related features, merged_df.shape is ', merged_df.shape)

merged_df.rename(index=str, 
                 columns={'treatment_acceleraion_add_value': 'treatment_acceleration_add_value', 
                          'reaserch_acceleration_add_value': 'research_acceleration_add_value',
                          'reaserch_acceleration_reduce_value': 'research_acceleration_reduce_value'}, 
                 inplace=True)
reduce_value_sum_ss, add_value_sum_ss = 0, 0
add_kind_ss, reduce_kind_ss = 0, 0
for name_t in ['wood', 'stone', 'ivory', 'meat', 
               'magic', 'infantry', 'cavalry', 'shaman',
               'wound_infantry', 'wound_cavalry', 'wound_shaman', 
               'general_acceleration', 'building_acceleration', 
               'research_acceleration', 'training_acceleration',
               'treatment_acceleration']: 
    merged_df[name_t+'_consume_ratio'] = merged_df[name_t+'_reduce_value'] / merged_df[name_t+'_add_value']
    reduce_value_sum_ss += merged_df[name_t+'_reduce_value']
    add_value_sum_ss += merged_df[name_t+'_add_value']
    add_kind_ss += np.where(merged_df[name_t+'_add_value']==0, 0, 1)
    reduce_kind_ss += np.where(merged_df[name_t+'_reduce_value']==0, 0, 1)
    
merged_df['all_consume_ratio'] = reduce_value_sum_ss / add_value_sum_ss
merged_df['all_add_count'] = add_kind_ss
merged_df['all_reduce_count'] = reduce_kind_ss
merged_df['all_add_value_sum'] = add_value_sum_ss
merged_df['all_reduce_value_sum'] = reduce_value_sum_ss


merged_df['pvp_lanch_ratio'] = merged_df['pvp_lanch_count'] / merged_df['pvp_battle_count']
merged_df['pvp_lanch_win_ratio'] = merged_df['pvp_win_count'] / merged_df['pvp_lanch_count']
merged_df['pvp_battle_win_ratio'] = merged_df['pvp_win_count'] / merged_df['pvp_battle_count']

merged_df['pve_lanch_ratio'] = merged_df['pve_lanch_count'] / merged_df['pve_battle_count']
merged_df['pve_lanch_win_ratio'] = merged_df['pve_win_count'] / merged_df['pve_lanch_count']
merged_df['pve_battle_win_ratio'] = merged_df['pve_win_count'] / merged_df['pve_battle_count']

merged_df['pay_price_avg'] = merged_df['pay_price'] / merged_df['pay_count']

print('with ratio related features, merged_df.shape is ', merged_df.shape)

# 上面除以0.0可能会得到np.inf的值
merged_df.replace(np.inf, 0, inplace=True)
merged_df.replace(np.nan, 0, inplace=True)
merged_df = merged_df.round(3)

merged_df.to_csv('C:/D_Disk/data_competition/gamer_value/data/merged_df.csv')
























