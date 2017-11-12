import numpy as np
np.random.seed(2017)

import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split


#city_shifted = login_sub_df['city'].shift(1)
#city_changed = login_sub_df['city']!=city_shifted
#login_sub_df[city_changed]


def get_one():
    global trade_df, merged_login_df
    risk_id = trade_df[trade_df.is_risk==1]['id'].unique()
    risk_login_df = merged_login_df[merged_login_df.id.isin(risk_id)]
    count_df = risk_login_df.groupby(risk_login_df['id']).count()
    print('count_df is ', count_df)

    
def get_change_city_min_elapse(login_df):
    min_elapse = 10**5
    for i in range(len(login_df)-1):
        ip, city, time = login_df.iloc[i][['ip', 'city', 'time']]
        ip_next, city_next, time_next = login_df.iloc[i+1][['ip', 'city', 'time']]
        if city!=city_next:
            elapse = (time_next - time)
            elapse = np.timedelta64(elapse, 's')
            elapse = elapse.astype('float')/60.0  # 计算间隔的分钟数，转换为float类型
            min_elapse = min(min_elapse, elapse)        
    return min_elapse
    

def get_neareast_N_login(N=20):  #得到最近20次登录切换城市的最小分钟数
    global trade_df, merged_login_df
    trade_df_new = trade_df.copy()
    min_elapses = []
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df.iterrows():
        print('compute ', count)
        count += 1
        trade_time = row['time']
        user_id = row['id']
        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        login_sub_df = login_sub_df.iloc[-N:, :]
        elaspe_time = get_change_city_min_elapse(login_sub_df)
        min_elapses.append(elaspe_time)
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_df_new['change_city_min_elaspe'] = np.array(min_elapses)
    trade_df_new.to_csv('change_city_min_elaspe.csv')
    
    
def get_city_ip_login_num():
    global trade_df, merged_login_df
    print('starting computing')
    start_t = time.time()
    trade_df_new = trade_df.copy()
    
    ip_count_df = merged_login_df[['ip']].groupby(merged_login_df['id']).unique().count()
    city_count_df = merged_login_df[['city']].groupby(merged_login_df['id']).unique().count()
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    
    trade_df_id_set = set(trade_df_new['id'])
    merged_login_df_id_set = set(ip_count_df.index)
    check = trade_df_id_set < merged_login_df_id_set
    print('check is ', check)
    
    trade_df_new['ip_num'] = ip_count_df['ip'][trade_df_new['id']].values
    trade_df_new['city_num'] = city_count_df['city'][trade_df_new['id']].values
    trade_df_new['city_num'] = city_count_df['city'][trade_df_new['id']].values
    trade_df_new.to_csv('ip_city_num.csv')
    

def compute_elaspe_time(time1, time2): #计算间隔时间，以分钟计算 会包含小数点部分
    if time1 > time2:
        time1, time2 = time2, time1
    elapse = (time2 - time1)
    elapse = np.timedelta64(elapse, 's')
    elapse = elapse.astype('float')/60.0  # 计算间隔的分钟数，转换为float类型
    return elapse
    

def compute_last_time(device, ip, city, last_login_time, login_df):
    device_t, ip_t, city_t = None, None, None
    
    device_t_series = login_df['time'][login_df.device==device]
    ip_t_series = login_df['time'][login_df.ip==ip]
    city_t_series = login_df['time'][login_df.city==city]
    if len(device_t_series)>0:
        device_t = device_t_series[-1]
    if len(ip_t_series)>0:
        ip_t = ip_t_series[-1]
    if len(city_t_series)>0:
        city_t = city_t_series[-1]

    result_t = []
    for t in (device_t, ip_t, city_t):
        time_t = 10.**6 if t==None else compute_elaspe_time(t, last_login_time)
        result_t.append(time_t)
    return result_t

    
def get_last_login_device_ip_city_time(): # 最后一次登录的device ip city所登录距离现在的时间（以分钟计算）如果从来没有登录过 则为10**6
    global trade_df, merged_login_df
    trade_df_new = trade_df.copy()
    device_new, ip_new, city_new = [], [], []
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df.iterrows():
        print('compute ', count)
        count += 1
        trade_time = row['time']
        user_id = row['id']
        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        if len(login_sub_df)==0:  # 如果没有最后一次登录,则为10.**6
            device_new.append[10.**6]
            ip_new.append[10.**6]
            city_new.append[10.**6]
        else:
            device, ip, city, last_login_t = login_sub_df.iloc[-1][['device', 'ip', 'city', 'time']]
            device_t, ip_t, city_t = compute_last_time(device, ip, city, last_login_t, login_sub_df.iloc[:-1])
            device_new.append(device_t)
            ip_new.append(ip_t)
            city_new.append(city_t)
        if count>10:
            break
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_df_new['device_new'] = np.array(device_new)
    trade_df_new['ip_new'] = np.array(ip_new)
    trade_df_new['city_new'] = np.array(city_new)
    trade_df_new.to_csv('last_login_device_ip_city_time.csv')
    

def get_last_half_year_city_ip_num():
    global trade_df, merged_login_df
    print('starting computing')
    start_t = time.time()
    trade_df_new = trade_df.copy()
    ip_count_df = merged_login_df[['ip']].groupby(merged_login_df['id']).unique().count()
    city_count_df = merged_login_df[['city']].groupby(merged_login_df['id']).unique().count()
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    
    trade_df_id_set = set(trade_df_new['id'])
    merged_login_df_id_set = set(ip_count_df.index)
    check = trade_df_id_set < merged_login_df_id_set
    print('check is ', check)
    
    trade_df_new['last_half_year_ip_num'] = ip_count_df['ip'][trade_df_new['id']].values
    trade_df_new['last_half_year_city_num'] = city_count_df['city'][trade_df_new['id']].values
    trade_df_new.to_csv('ip_city_num.csv')
    
    
    
if __name__=='__main__':
    start_t = time.time()
    trade_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade.csv', 
                           index_col='rowkey', dtype={'id': np.str})
    trade_test_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade_test.csv', 
                                index_col='rowkey', dtype={'id': np.str})
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    login_df = pd.read_csv('./data/Risk_Detection_Qualification/t_login.csv', 
                           index_col='log_id', 
                           dtype={'id': np.str, 'timestamp': np.str},
                           parse_dates=['time'], date_parser=dateparse)
    login_test_df = pd.read_csv('./data/Risk_Detection_Qualification/t_login_test.csv', 
                                index_col='log_id', 
                                dtype={'id': np.str, 'timestamp': np.str},
                                parse_dates=['time'], date_parser=dateparse)
    merged_login_df = login_df.append(login_test_df)
    end_t = time.time()
    print('load cost time: ', end_t-start_t)
    
#    get_neareast_N_login()
    get_city_ip_num()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    