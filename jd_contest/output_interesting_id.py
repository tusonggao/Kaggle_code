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
    
#------------------------------------------------------------------------------------#
#得到本次交易最近20次登录切换城市的最小分钟数
def get_neareast_N_change_city_min_elaspe(N=20):  
    global trade_df, merged_login_df
    trade_df_new = trade_df.copy()
    min_elapses = []
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df.iterrows():
        print('get_neareast_N_change_city_min_elaspe compute ', count)
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
    trade_df_new.to_csv('./features/change_city_min_elaspe.csv')    

def get_change_city_min_elapse(login_df):
    min_elapse = 10.**8
    for i in range(len(login_df)-1):
        ip, city, time = login_df.iloc[i][['ip', 'city', 'time']]
        ip_next, city_next, time_next = login_df.iloc[i+1][['ip', 'city', 'time']]
        if city!=city_next:
            elapse = (time_next - time)
            elapse = np.timedelta64(elapse, 's')
            elapse = elapse.astype('float')/60.0  # 计算间隔的分钟数，转换为float类型
            min_elapse = min(min_elapse, elapse)        
    return min_elapse
    
#------------------------------------------------------------------------------------#
#本次交易最后一次登录的device ip city所登录距离现在的时间（以分钟计算）如果从来没有登录过 则为10**6
def get_last_login_device_ip_city_time(): 
    global trade_df, merged_login_df
    trade_df_new = trade_df.copy()
    device_new, ip_new, city_new = [], [], []
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df.iterrows():
        print('get_last_login_device_ip_city_time compute ', count)
        count += 1
        trade_time = row['time']
        user_id = row['id']
        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        if len(login_sub_df)==0:  # 如果没有最后一次登录,则为10.**6
            device_new.append(10.**8)
            ip_new.append(10.**8)
            city_new.append(10.**8)
        else:
            device, ip, city = login_sub_df.iloc[-1][['device', 'ip', 'city']]
            last_login_t = login_sub_df['time'].values[-1]
            device_t, ip_t, city_t = compute_last_time(device, ip, city, last_login_t, login_sub_df.iloc[:-1])
            device_new.append(device_t)
            ip_new.append(ip_t)
            city_new.append(city_t)
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_df_new['device_new'] = np.array(device_new)
    trade_df_new['ip_new'] = np.array(ip_new)
    trade_df_new['city_new'] = np.array(city_new)
    trade_df_new.to_csv('last_login_device_ip_city_time.csv')
    
def compute_last_time(device, ip, city, last_login_time, login_df):
    device_t, ip_t, city_t = None, None, None
    
    device_t_series = login_df['time'][login_df.device==device]
    ip_t_series = login_df['time'][login_df.ip==ip]
    city_t_series = login_df['time'][login_df.city==city]
    if len(device_t_series)>0:
        device_t = device_t_series.values[-1]
    if len(ip_t_series)>0:
        ip_t = ip_t_series.values[-1]
    if len(city_t_series)>0:
        city_t = city_t_series.values[-1]

    result_t = []
    for t in (device_t, ip_t, city_t):
        if t==None:
            time_t = 10.**6 
        else:
            time_t = compute_elaspe_time(t, last_login_time)
        result_t.append(time_t)
    return result_t

#计算间隔时间，以分钟计算 会包含小数点部分 time1 time2 为 np.datetime64 类型
def compute_elaspe_time(time1, time2):
    if isinstance(time1, str):
        time1 = np.datetime64(time1)
    if isinstance(time2, str):
        time2 = np.datetime64(time2)
    if time1 > time2:
        time1, time2 = time2, time1
    elapse = (time2 - time1)
    elapse = np.timedelta64(elapse, 's')
    elapse = elapse.astype('float')/60.0  # 计算间隔的分钟数，转换为float类型
    return elapse

#------------------------------------------------------------------------------------#
#本次交易是否是当天第一次交易，是否是当天第一次登录
def get_whether_today_first_trade_login():  
    global trade_df, merged_login_df
    trade_df_new = trade_df.copy()
    trade_df_new = trade_df_new.sort_values(by='time')
    first_trade_list, first_login_list = [], []
    id_newest_trade_time = {}
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df_new.iterrows():
        print('get_whether_today_first_trade_login compute ', count)
        count += 1
        trade_time = row['time']
        user_id = row['id']
        if user_id not in id_newest_trade_time:
            first_trade_list.append(1)
        else:
            last_trade_time = id_newest_trade_time[user_id]
            if check_whether_same_day(trade_time, last_trade_time):
                first_trade_list.append(0)
            else:
                first_trade_list.append(1)
        id_newest_trade_time[user_id] = trade_time

        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        if login_sub_df.shape[0]<2:
            first_login_list.append(1)
        else:
            login_t = login_sub_df['time'].values[-1]
            last_login_t = login_sub_df['time'].values[-2]
            if check_whether_same_day(login_t, last_login_t):
                first_login_list.append(0)
            else:
                first_login_list.append(1)
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_df_new['today_first_trade'] = np.array(first_trade_list)
    trade_df_new['today_first_login'] = np.array(first_login_list)
    trade_df_new.to_csv('./features/whether_today_first_trade_login.csv')

#time1 time2为 np.datetime64 类型
def check_whether_same_day(time1, time2):
    time1 = pd.DatetimeIndex([time1])
    time2 = pd.DatetimeIndex([time2])
    year1, month1, day1 = time1.year[0], time1.month[0], time1.day[0]
    year2, month2, day2 = time2.year[0], time2.month[0], time2.day[0]
    same_day = (year1==year2 and month1==month2 and day1==day2)
    return 1 if same_day else 0

    
#------------------------------------------------------------------------------------#
#本次交易距离上一次交易时间，本次登录距离上一次登录时间，本次交易与登录之间的时间差，以分钟计数
def get_last_login_trade_time_elapse():  
    global trade_df, merged_login_df
    trade_df_new = trade_df.copy()
    trade_df_new = trade_df_new.sort_values(by='time')
    trade_df_new = trade_df_new.iloc[:200]
    trade_elapse_list, login_elapse_list = [], []
    id_newest_trade_time = {}
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df_new.iterrows():
        print('get_whether_today_first_trade_login compute ', count)
        count += 1
        trade_time = row['time']
        print('type of trade_time is ', type(trade_time))
        user_id = row['id']
        if user_id not in id_newest_trade_time:
            trade_elapse_list.append(10.**6)
        else:
            last_trade_time = id_newest_trade_time[user_id]
            print('type of last_trade_time is ', type(last_trade_time))
            elapse_t = compute_elaspe_time(trade_time, last_trade_time)
            trade_elapse_list.append(elapse_t)
        id_newest_trade_time[user_id] = trade_time

        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        if login_sub_df.shape[0]<2:
            login_elapse_list.append(10.**6)
        else:
            login_t = login_sub_df['time'].values[-1]
            last_login_t = login_sub_df['time'].values[-2]
            elapse_t = compute_elaspe_time(login_t, last_login_t)
            login_elapse_list.append(elapse_t)
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_df_new['last_trade_elapse'] = np.array(trade_elapse_list)
    trade_df_new['last_login_elapse'] = np.array(login_elapse_list)
    trade_df_new.to_csv('./features/last_login_trade_time_elapse.csv')

#------------------------------------------------------------------------------------#
#得本次交易登录的device ip city是否与上一次登录相同
def get_whether_this_trade_new_device_ip_city():  
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
    

#------------------------------------------------------------------------------------#
#得到该id在这之前的总交易次数、总登录次数、登录过的city数、ip数、device数
def get_before_trade_login_city_ip_device_num():  
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
#    get_city_ip_num()
#    get_last_login_device_ip_city_time()
#    get_whether_today_first_trade_login()
    get_last_login_trade_time_elapse()
    
    