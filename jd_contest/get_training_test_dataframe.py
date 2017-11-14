import numpy as np
np.random.seed(2017)

import time
import pandas as pd
import matplotlib.pyplot as plt

def add_features(data_df, features_dir):
    df = pd.read_csv(features_dir + 'change_city_min_elaspe.csv',
                     index_col='rowkey', dtype={'id': np.str})
    data_df['change_city_min_elaspe'] = df['change_city_min_elaspe']


    df = pd.read_csv(features_dir + 'last_login_device_ip_city_elapse.csv',
                     index_col='rowkey', dtype={'id': np.str})
    data_df[['last_device_elapse', 'last_ip_elapse', 'last_city_elapse']] = df[['last_device_elapse', 'last_ip_elapse', 'last_city_elapse']]

    df = pd.read_csv(features_dir + 'whether_today_first_trade_login.csv',
                     index_col='rowkey', dtype={'id': np.str})
    data_df[['today_first_trade', 'today_first_login']] = df[['today_first_trade', 'today_first_login']]


    df = pd.read_csv(features_dir + 'last_login_trade_time_elapse.csv',
                     index_col='rowkey', dtype={'id': np.str})
    data_df[['last_trade_elapse', 'last_login_elapse', 'trade_login_elapse']] = df[['last_trade_elapse', 'last_login_elapse', 'trade_login_elapse']]


    df = pd.read_csv(features_dir + 'whether_this_trade_new_device_ip_city.csv',
                     index_col='rowkey', dtype={'id': np.str})
    data_df[['same_device', 'same_ip', 'same_city']] = df[['same_device', 'same_ip', 'same_city']]

    df = pd.read_csv(features_dir + 'last3_login_info.csv',
                     index_col='rowkey', dtype={'id': np.str})
    data_df[['time_long1', 'time_long2', 'time_long3']] = df[['time_long1', 'time_long2', 'time_long3']]
    data_df[['log_from1', 'log_from2', 'log_from3']] = df[['log_from1', 'log_from2', 'log_from3']]
    data_df[['result1', 'result2', 'result3']] = df[['result1', 'result2', 'result3']]
    data_df[['is_scan1', 'is_scan2', 'is_scan3']] = df[['is_scan1', 'is_scan2', 'is_scan3']]

    df = pd.read_csv(features_dir + 'before_trade_trade_login_num.csv',
                     index_col='rowkey', dtype={'id': np.str})
    data_df[['before_trade_num', 'before_login_num']] = df[['before_trade_num', 'before_login_num']]

    return data_df

def generate_training_dataframe():
    global trade_df
    train_df_new = trade_df.copy()
    train_df_new = train_df_new.iloc[:, :-1]
    features_dir = './features/train/'
    return add_features(train_df_new, features_dir)
    
def generate_test_dataframe():
    global trade_test_df
    features_dir = './features/test/'
    test_df_new = trade_test_df.copy()
    return add_features(test_df_new, features_dir)

def expand_df(merged_df):
    fill_na_columns = ['change_city_min_elaspe', 'last_device_elapse', 
                       'last_ip_elapse', 'last_city_elapse', 
                       'today_first_trade', 'today_first_login',
                       'last_trade_elapse', 'last_login_elapse', 
                       'trade_login_elapse','time_long1', 
                       'time_long2', 'time_long3']
    for col in fill_na_columns:
        merged_df[col].fillna(merged_df[col].mean(), inplace=True)

    merged_df[['log_from1', 'log_from2', 'log_from3']] =merged_df[['log_from1', 'log_from2', 'log_from3']].astype('str')
    merged_df[['result1', 'result2', 'result3']] =merged_df[['result1', 'result2', 'result3']].astype('str')
    merged_df[['is_scan1', 'is_scan2', 'is_scan3']] =merged_df[['is_scan1', 'is_scan2', 'is_scan3']].astype('str')
    
    merged_df.drop('time', axis=1, inplace=True)
    merged_df = pd.get_dummies(merged_df)
        
    return merged_df
    
    
if __name__=='__main__':
    start_t = time.time()
    trade_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade.csv', 
                           index_col='rowkey')
    trade_test_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade_test.csv', 
                                index_col='rowkey')
    
    print('trade_df dtypes is ', trade_df.dtypes)
    
    end_t = time.time()
    print('load cost time: ', end_t-start_t)
    
    train_num = len(trade_df)
    train_y = trade_df['is_risk']
    
    training_df = generate_training_dataframe()
    test_df = generate_test_dataframe()
    
    merged_df = training_df.append(test_df)
    merged_df = expand_df(merged_df)
    train_df = pd.concat([merged_df.iloc[:train_num], train_y], axis=1)
    test_df = merged_df.iloc[train_num:]

    train_df.to_csv('./data/train_data.csv')
    test_df.to_csv('./data/test_data.csv')
    
    end_t = time.time()
    print('total cost time: ', end_t-start_t)
    
    
    
    
    
    
    
#    df = pd.read_csv('./features/test/last3_login_info.csv')
#    print('df is ', df.count())
#    result_df = pd.get_dummies(df['result1'])
#    print(result_df.head(20))
    
#    df = pd.DataFrame({'A': [1, 2, 3, np.nan, 7], 'B': [-4.1, 5.3, 2, np.nan, 4], 
#                       'C': [100, 300, 900, np.nan, 200]})
#    
#    df_1 = pd.DataFrame({'D': [10, 20, 30, np.nan, 70]})
#    
#    df_new = pd.concat([df, df_1], axis=1)
#    print('df_new is ', df_new)

#    df[['A', 'B', 'C']].fillna(df[['A', 'B', 'C']].mean(), inplace=True)
#    df['A'].fillna(df['A'].mean(), inplace=True)
#    print(df)
#    df[['B', 'C']] = df[['B', 'C']].astype('str')
#    df_dummies = pd.get_dummies(df)
#    print(df_dummies)
    
#    df.to_csv('test_output111.csv')
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    