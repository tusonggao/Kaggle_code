import numpy as np
np.random.seed(2017)

import time
import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split

#------------------------------------------------------------------------------------#
#去除重复的登录记录 如果前后相差不到3分钟 device log_from ip city result id type is_scan都相同
#则去掉后面这条记录
def get_sameday_sameid_different_trade_risk(trade_df):
    result_df = pd.DataFrame()
    id_list = sorted(login_df['id'].unique())
    print('len of id_list is ', len(id_list))
    count = 0
    for user_id in id_list:
        count += 1
        print('processing user_id ', user_id)
        sub_trade_df = trade_df[trade_df['id']==user_id].sort_values(by='time', ascending = True)
        row_num = 0
        question_row_num_list = []
        while row_num < len(sub_trade_df)-1:
            rec1 = sub_trade_df.iloc[row_num]
            rec2 = sub_trade_df.iloc[row_num+1]
            if (compute_elaspe_time(rec1['time'], rec2['time'])<=24*60.0 and
                rec1['is_risk']!=rec2['is_risk']):
                question_row_num_list.append(row_num)
            row_num += 1
        question_sub_trade_df = sub_trade_df.iloc[question_row_num_list]
        result_df = result_df.append(question_sub_trade_df)
#        if count >=20:
#            break
    result_df.to_csv('./data/sameday_sameid_different_trade_risk.csv')
    return result_df


#------------------------------------------------------------------------------------#
#去除重复的登录记录 如果前后相差不到3分钟 device log_from ip city result id type is_scan都相同
#则去掉后面这条记录
def is_duplicate_records(rec1, rec2):
    check_fields = ['device', 'log_from', 'ip', 'city', 'result', 'type']
    for field in check_fields:
        if rec1[field] != rec2[field]:
            return False
    return compute_elaspe_time(rec1['time'], rec2['time']) <= 3.0 # 小于3分钟

def remove_duplicate_login_records(login_df):
    result_df = pd.DataFrame()
    id_list = sorted(login_df['id'].unique())
    print('len of id_list is ', len(id_list))
    count = 0
    for user_id in id_list:
        count += 1
        print('processing user_id ', user_id)
        sub_login_df = login_df[login_df['id']==user_id].sort_values(by='time', ascending = True)
        removed_sub_login_df = pd.DataFrame()
        row_num = 0
        valid_row_num_list = [0]
        while row_num < len(sub_login_df)-1:
            if not is_duplicate_records(sub_login_df.iloc[row_num], 
                                        sub_login_df.iloc[row_num+1]):
                valid_row_num_list.append(row_num+1)
            else:
                print('skipped one row user_id row_num is ', user_id, row_num+1)
            row_num += 1
        removed_sub_login_df = sub_login_df.iloc[valid_row_num_list]
        result_df = result_df.append(removed_sub_login_df)
        if count >=100:
            break
    result_df.to_csv('./data/removed_duplicate.csv')
    return result_df


    
#------------------------------------------------------------------------------------#
#得到本次交易最近20次登录切换城市的最小分钟数
def get_neareast_N_change_city_min_elaspe(N=20):  
    global trade_df, merged_login_df, outputdir
    trade_df_new = trade_df.copy()
#    trade_df_new = trade_df_new.iloc[:100]
    min_elapses = []
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df_new.iterrows():
        print('get_neareast_N_change_city_min_elaspe compute ', count)
        count += 1
        trade_time = row['time']
        user_id = row['id']
        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        login_sub_df = login_sub_df.iloc[-N:, :]
        elaspe_time = get_change_city_min_elapse(login_sub_df)
        if elaspe_time==10.**8:
            min_elapses.append(np.nan)
        else:            
            min_elapses.append(elaspe_time)
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_df_new['change_city_min_elaspe'] = np.array(min_elapses)
    trade_df_new.to_csv(outputdir + 'change_city_min_elaspe.csv')

    
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
#本次交易最后一次登录的device ip city所登录距离现在的时间（以分钟计算）如果从来没有登录过 则为10**8
def get_last_login_device_ip_city_elapse(): 
    global trade_df, merged_login_df, outputdir
    trade_df_new = trade_df.copy()
#    trade_df_new = trade_df_new.iloc[:100]
    device_new, ip_new, city_new = [], [], []
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df_new.iterrows():
        print('get_last_login_device_ip_city_elapse compute ', count)
        count += 1
        trade_time = row['time']
        user_id = row['id']
        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        if len(login_sub_df)==0:
            device_new.append(np.nan)
            ip_new.append(np.nan)
            city_new.append(np.nan)
        else:
            device, ip, city = login_sub_df.iloc[-1][['device', 'ip', 'city']]
            last_login_t = login_sub_df['time'].values[-1]
            device_t, ip_t, city_t = compute_last_time(device, ip, city, last_login_t, login_sub_df.iloc[:-1])
            device_new.append(device_t)
            ip_new.append(ip_t)
            city_new.append(city_t)
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_df_new['last_device_elapse'] = np.array(device_new)
    trade_df_new['last_ip_elapse'] = np.array(ip_new)
    trade_df_new['last_city_elapse'] = np.array(city_new)
    trade_df_new.to_csv(outputdir + 'last_login_device_ip_city_elapse.csv')
    
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
            time_t = np.nan
        else:
            time_t = compute_elaspe_time(t, last_login_time)
        result_t.append(time_t)
    return result_t

#计算间隔时间，以分钟计算 会包含小数点部分 time1 time2 为 np.datetime64 类型
def compute_elaspe_time(time1, time2):
    if not isinstance(time1, np.datetime64):
        time1 = np.datetime64(time1)
    if not isinstance(time2, np.datetime64):
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
    global trade_df, merged_login_df, outputdir
    trade_df_new = trade_df.copy()
    trade_df_new = trade_df_new.sort_values(by='time')
#    trade_df_new = trade_df_new.iloc[:100]
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
    trade_df_new.to_csv(outputdir + 'whether_today_first_trade_login.csv')

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
    global trade_df, merged_login_df, outputdir
    trade_df_new = trade_df.copy()
    trade_df_new = trade_df_new.sort_values(by='time')
#    trade_df_new = trade_df_new.iloc[:100]
    trade_elapse_list, login_elapse_list, trade_login_elapse_list = [], [], []
    id_newest_trade_time = {}
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df_new.iterrows():
        print('get_last_login_trade_time_elapse compute ', count)
        count += 1
        trade_time = row['time']
        user_id = row['id']
        if user_id not in id_newest_trade_time:
            trade_elapse_list.append(np.nan)
        else:
            last_trade_time = id_newest_trade_time[user_id]
            elapse_t = compute_elaspe_time(trade_time, last_trade_time)
            trade_elapse_list.append(elapse_t)
        id_newest_trade_time[user_id] = trade_time

        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        if login_sub_df.shape[0]>=1:
            login_t = login_sub_df['time'].values[-1]
#            print('type of login_t and trade_time is ', type(login_t), type(trade_time))
            elapse_t = compute_elaspe_time(login_t, trade_time)
            trade_login_elapse_list.append(elapse_t)
            if login_sub_df.shape[0]>1:
                last_login_t = login_sub_df['time'].values[-2]
                elapse_t = compute_elaspe_time(login_t, last_login_t)
                login_elapse_list.append(elapse_t)     
            else:
                login_elapse_list.append(np.nan)     
        else:
            login_elapse_list.append(np.nan)
            trade_login_elapse_list.append(np.nan)
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_df_new['last_trade_elapse'] = np.array(trade_elapse_list)
    trade_df_new['last_login_elapse'] = np.array(login_elapse_list)
    trade_df_new['trade_login_elapse'] = np.array(trade_login_elapse_list)
    trade_df_new.to_csv(outputdir + 'last_login_trade_time_elapse.csv')

#------------------------------------------------------------------------------------#
#得本次交易登录的device ip city是否与上一次登录相同，如果没有上次登录 默认为0
def get_whether_this_trade_same_device_ip_city():  
    global trade_df, merged_login_df, outputdir
    trade_df_new = trade_df.copy()
    trade_df_new = trade_df_new.sort_values(by='time')
#    trade_df_new = trade_df_new.iloc[:100]
    same_device_list, same_ip_list, same_city_list = [], [], []
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df_new.iterrows():
        print('get_whether_this_trade_same_device_ip_city compute ', count)
        count += 1
        trade_time = row['time']
        user_id = row['id']
        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        if login_sub_df.shape[0]>=2:
            device1, device2 = list(login_sub_df['device'].values[-2:])
            ip1, ip2 = list(login_sub_df['ip'].values[-2:])
            city1, city2 = list(login_sub_df['city'].values[-2:])
            same_device_list.append(1 if device1==device2 else 0)
            same_ip_list.append(1 if device1==device2 else 0)
            same_city_list.append(1 if device1==device2 else 0)
        else:
            same_device_list.append(0)
            same_ip_list.append(0)
            same_city_list.append(0)
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_df_new['same_device'] = np.array(same_device_list)
    trade_df_new['same_ip'] = np.array(same_ip_list)
    trade_df_new['same_city'] = np.array(same_city_list)
    trade_df_new.to_csv(outputdir + 'whether_this_trade_new_device_ip_city.csv')
    
#------------------------------------------------------------------------------------#
#得本次交易最近三次登录的登录时间、登录来源、登录结果、是否扫码
def get_last3_login_info():  
    global trade_df, merged_login_df, outputdir
    trade_df_new = trade_df.copy()
    trade_df_new = trade_df_new.sort_values(by='time')
#    trade_df_new = trade_df_new.iloc[:100]
    
    time_long1_list, time_long2_list, time_long3_list = [], [], []
    log_from1_list, log_from2_list, log_from3_list = [], [], []
    result1_list, result2_list, result3_list = [], [], []
    is_scan1_list, is_scan2_list, is_scan3_list = [], [], []
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df_new.iterrows():
        print('get_last3_login_info compute ', count)
        count += 1
        trade_time = row['time']
        user_id = row['id']
        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        if login_sub_df.shape[0]>=3:
            time_long1_list.append(login_sub_df['timelong'].values[-1])
            time_long2_list.append(login_sub_df['timelong'].values[-2])
            time_long3_list.append(login_sub_df['timelong'].values[-3])
            log_from1_list.append(login_sub_df['log_from'].values[-1])
            log_from2_list.append(login_sub_df['log_from'].values[-2])
            log_from3_list.append(login_sub_df['log_from'].values[-3])
            result1_list.append(login_sub_df['result'].values[-1])
            result2_list.append(login_sub_df['result'].values[-2])
            result3_list.append(login_sub_df['result'].values[-3])
            is_scan1_list.append(login_sub_df['is_scan'].values[-1])
            is_scan2_list.append(login_sub_df['is_scan'].values[-2])
            is_scan3_list.append(login_sub_df['is_scan'].values[-3])
        elif login_sub_df.shape[0]==2:
            time_long1_list.append(login_sub_df['timelong'].values[-1])
            time_long2_list.append(login_sub_df['timelong'].values[-2])
            time_long3_list.append(np.nan)
            log_from1_list.append(login_sub_df['log_from'].values[-1])
            log_from2_list.append(login_sub_df['log_from'].values[-2])
            log_from3_list.append(np.nan)
            result1_list.append(login_sub_df['result'].values[-1])
            result2_list.append(login_sub_df['result'].values[-2])
            result3_list.append(np.nan)
            is_scan1_list.append(login_sub_df['is_scan'].values[-1])
            is_scan2_list.append(login_sub_df['is_scan'].values[-2])
            is_scan3_list.append(np.nan)
        elif login_sub_df.shape[0]==1:
            time_long1_list.append(login_sub_df['timelong'].values[-1])
            time_long2_list.append(np.nan)
            time_long3_list.append(np.nan)
            log_from1_list.append(login_sub_df['log_from'].values[-1])
            log_from2_list.append(np.nan)
            log_from3_list.append(np.nan)
            result1_list.append(login_sub_df['result'].values[-1])
            result2_list.append(np.nan)
            result3_list.append(np.nan)
            is_scan1_list.append(login_sub_df['is_scan'].values[-1])
            is_scan2_list.append(np.nan)
            is_scan3_list.append(np.nan)
        else:
            time_long1_list.append(np.nan)
            time_long2_list.append(np.nan)
            time_long3_list.append(np.nan)
            log_from1_list.append(np.nan)
            log_from2_list.append(np.nan)
            log_from3_list.append(np.nan)
            result1_list.append(np.nan)
            result2_list.append(np.nan)
            result3_list.append(np.nan)
            is_scan1_list.append(np.nan)
            is_scan2_list.append(np.nan)
            is_scan3_list.append(np.nan)
            
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_df_new['time_long1'] = np.array(time_long1_list)
    trade_df_new['time_long2'] = np.array(time_long2_list)
    trade_df_new['time_long3'] = np.array(time_long3_list)
    
    trade_df_new['log_from1'] = np.array(log_from1_list)
    trade_df_new['log_from2'] = np.array(log_from2_list)
    trade_df_new['log_from3'] = np.array(log_from3_list)
    
    trade_df_new['result1'] = np.array(result1_list)
    trade_df_new['result2'] = np.array(result2_list)
    trade_df_new['result3'] = np.array(result3_list)
    
    trade_df_new['is_scan1'] = np.array(is_scan1_list)
    trade_df_new['is_scan2'] = np.array(is_scan2_list)
    trade_df_new['is_scan3'] = np.array(is_scan3_list)
    trade_df_new.to_csv(outputdir + 'last3_login_info.csv')
    
    
#------------------------------------------------------------------------------------#
#得到该id在这之前的总交易次数、总登录次数
def get_before_trade_trade_login_num():  
    global trade_df, merged_login_df, outputdir
    trade_df_new = trade_df.copy()
    trade_df_new = trade_df_new.sort_values(by='time')
#    trade_df_new = trade_df_new.iloc[:100]
    trade_num_list, login_num_list = [], []
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df_new.iterrows():
        print('get_before_trade_trade_login_num compute ', count)
        count += 1
        trade_time = row['time']
        user_id = row['id']
        trade_sub_df = trade_df_new[trade_df_new.id==user_id]
        trade_sub_df = trade_sub_df[trade_sub_df.time<=trade_time]
        trade_num_list.append(len(trade_sub_df))

        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        login_num_list.append(len(login_sub_df))
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_df_new['before_trade_num'] = np.array(trade_num_list)
    trade_df_new['before_login_num'] = np.array(login_num_list)
    trade_df_new.to_csv(outputdir + 'before_trade_trade_login_num.csv')   
    
#------------------------------------------------------------------------------------#
#得到本次交易距离2015-01-01 00:00:00的分钟数
def get_from_2015_1_1_minutes_num():
    global trade_df, outputdir
    trade_df_new = trade_df.copy()
    trade_df_new['from_2015_1_1_minutes_num'] = trade_df['time'].apply(
        lambda x: compute_elaspe_time(x, np.datetime64('2015-01-01 00:00:00')))
    trade_df_new.to_csv(outputdir + 'from_2015_1_1_minutes_num.csv')

#------------------------------------------------------------------------------------#
#得到该id在这本次交易和最后一次登录之间的总交易次数
def get_from_last_login_trade_num():  
    global trade_df, merged_login_df, outputdir
    trade_df_new = trade_df.copy()
    trade_df_new = trade_df_new.sort_values(by='time')
#    trade_df_new = trade_df_new.iloc[:100]
    trade_num_list= []
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df_new.iterrows():
        print('get_from_last_login_trade_num compute ', count)
        count += 1
        trade_time = row['time']
        user_id = row['id']
        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        if login_sub_df.shape[0]>0:
            last_login_dt = login_sub_df['time'].values[-1]
        else:
            last_login_dt = np.datetime64('2015-01-01 00:00:00')
        trade_sub_df = trade_df_new[trade_df_new.id==user_id]
        trade_sub_df = trade_sub_df[trade_sub_df.time<trade_time]
        trade_sub_df = trade_sub_df[trade_sub_df.time>=last_login_dt]
        trade_num_list.append(trade_sub_df.shape[0])
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_df_new['from_last_login_trade_num'] = np.array(trade_num_list)
    trade_df_new.to_csv(outputdir + 'from_last_login_trade_num.csv')
    
#------------------------------------------------------------------------------------#
#得到本次交易是否发生在凌晨1点到7点之间，如果是为1，否则为0
def get_whether_between_1_and_7_am():
    start_t = time.time()
    global trade_df, outputdir
    trade_df_new = trade_df.copy()
    trade_df_new['between_1_and_7_am'] = trade_df['time'].apply(
         lambda x: is_between_1_and_7_am(x))
    end_t = time.time()
    trade_df_new.to_csv(outputdir + 'whether_between_1_and_7_am.csv')
    print('get_whether_between_1_and_7_am cost time is ', end_t-start_t)
    

def is_between_1_and_7_am(dt):
    dt_index = pd.DatetimeIndex([dt])
    year, month, day = dt_index.year[0], dt_index.month[0], dt_index.day[0]
    dt_start = np.datetime64('%04d-%02d-%02d 01:00:00'%(year, month, day))
    dt_end = np.datetime64('%04d-%02d-%02d 07:00:00'%(year, month, day))
    return dt>=dt_start and dt<=dt_end
    

def check_835072_device():
    global trade_df, merged_login_df, outputdir
    trade_df_new = trade_df.copy()
    trade_df_new = trade_df_new.sort_values(by='time')
#    trade_df_new = trade_df_new.iloc[:100]
    check_list= []
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df_new.iterrows():
        print('check_835072_device compute ', count)
        count += 1
        trade_time = row['time']
        user_id = row['id']
        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        check_status = False
        if login_sub_df.shape[0]>0:
            last_login_device = login_sub_df['device'].values[-1]
            if last_login_device==835072:
                check_status = True
        check_list.append(check_status)
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_sub_df = trade_df_new[np.array(check_list)]
    trade_sub_df.to_csv(outputdir + 'check_835072_device.csv')

def check_different_device_ip_city():
    global trade_df, merged_login_df, outputdir
    trade_df_new = trade_df.copy()
    trade_df_new = trade_df_new.sort_values(by='time')
#    trade_df_new = trade_df_new.iloc[:100]
    check_list= []
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df_new.iterrows():
        print('check_different_device_ip_city compute ', count)
        count += 1
        trade_time = row['time']
        user_id = row['id']
        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        check_status = False
        if login_sub_df.shape[0]>1:
            login_device = login_sub_df['device'].values[-1]
            last_login_device = login_sub_df['device'].values[-2]
            login_ip = login_sub_df['ip'].values[-1]
            last_login_ip = login_sub_df['ip'].values[-2]
            login_city = login_sub_df['city'].values[-1]
            last_login_city = login_sub_df['city'].values[-2]
            login_result = login_sub_df['result'].values[-1]
            if (login_device!=last_login_device and login_ip!=last_login_ip 
                and login_city!=last_login_city and login_result==-2):
                check_status = True
        check_list.append(check_status)
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_sub_df = trade_df_new[np.array(check_list)]
    trade_sub_df.to_csv(outputdir + 'different_device_ip_city_with_result.csv')
    
#------------------------------------------------------------------------------------#
#截止到本次交易前该id最近一天内的总交易次数，总登录次数，总交易次数/(总登录次数+1)比值
#截止到本次交易前该id最近三天内的总交易次数，总登录次数，总交易次数/(总登录次数+1)比值
#截止到本次交易前该id最近七天内的总交易次数，总登录次数，总交易次数/(总登录次数+1)比值
#截止到本次交易前该id最近30天内的总交易次数，总登录次数，总交易次数/(总登录次数+1)比值
#截止到本次交易前该id（2015.1.1到今天）总交易次数，总登录次数，总交易次数/(总登录次数+1)比值
def get_till_now_login_trade_num():
    global trade_df, merged_login_df, outputdir
    trade_df_new = trade_df.copy()
    trade_df_new = trade_df_new.sort_values(by='time')  # trade_df_new = trade_df_new.iloc[:100]
    trade_num_list_1day, login_num_list_1day = [], []
    trade_num_list_3day, login_num_list_3day = [], []
    trade_num_list_7day, login_num_list_7day = [], []
    trade_num_list_30day, login_num_list_30day = [], []
    trade_num_list_history, login_num_list_history = [], []
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df_new.iterrows():
        print('get_from_last_login_trade_num compute ', count)
        count += 1
        trade_time, user_id = row['time'], row['id']
        trade_sub_df = trade_df_new[trade_df_new.id==user_id]
        login_sub_df = merged_login_df[merged_login_df.id==user_id]
        day_num_list = [1, 3, 7, 30, 1000]
        login_num_lists = [login_num_list_1day, login_num_list_3day,
                           login_num_list_7day, login_num_list_30day,
                           login_num_list_history]
        trade_num_lists = [trade_num_list_1day, trade_num_list_3day,
                           trade_num_list_7day, trade_num_list_30day,
                           trade_num_list_history]
        for num, login_l, trade_l in zip(day_num_list, login_num_lists, trade_num_lists):
            login_n, trade_n = handle_num_day(num, trade_time, login_sub_df, trade_sub_df)
            login_l.append(login_n)
            trade_l.append(trade_n)
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    
    trade_df_new['till_now_1day_login_num'] = np.array(login_num_list_1day)
    trade_df_new['till_now_1day_trade_num'] = np.array(trade_num_list_1day)
    trade_df_new['till_now_1day_trade_login_ratio'] = trade_df_new['till_now_1day_trade_num']/(trade_df_new['till_now_1day_login_num'] + 1)
    
    trade_df_new['till_now_3day_login_num'] = np.array(login_num_list_3day)
    trade_df_new['till_now_3day_trade_num'] = np.array(trade_num_list_3day)
    trade_df_new['till_now_3day_trade_login_ratio'] = trade_df_new['till_now_3day_trade_num']/(trade_df_new['till_now_3day_login_num'] + 1)
    
    trade_df_new['till_now_7day_login_num'] = np.array(login_num_list_7day)
    trade_df_new['till_now_7day_trade_num'] = np.array(trade_num_list_7day)
    trade_df_new['till_now_7day_trade_login_ratio'] = trade_df_new['till_now_7day_trade_num']/(trade_df_new['till_now_7day_login_num'] + 1)
    
    trade_df_new['till_now_30day_login_num'] = np.array(login_num_list_30day)
    trade_df_new['till_now_30day_trade_num'] = np.array(trade_num_list_30day)
    trade_df_new['till_now_30day_trade_login_ratio'] = trade_df_new['till_now_30day_trade_num']/(trade_df_new['till_now_30day_login_num'] + 1)
    
    trade_df_new['till_now_history_login_num'] = np.array(login_num_list_history)
    trade_df_new['till_now_history_trade_num'] = np.array(trade_num_list_history)
    trade_df_new['till_now_history_trade_login_ratio'] = trade_df_new['till_now_history_trade_num']/(trade_df_new['till_now_history_login_num'] + 1)
    
    trade_df_new.to_csv(outputdir + 'till_now_login_trade_num(.csv')

def handle_num_day(day_num, trade_time, login_sub_df, trade_sub_df):
    early_t = trade_time - np.timedelta64(day_num, 'D')
    earlier_than_login = login_sub_df.time <= trade_time
    later_than_login = login_sub_df.time > early_t
    earlier_than_trade = trade_sub_df.time <= trade_time
    later_than_trade = trade_sub_df.time > early_t
    return (len(login_sub_df[earlier_than_login & later_than_login]),
            len(trade_sub_df[earlier_than_trade & later_than_trade]))
    
    
#------------------------------------------------------------------------------------#
#得到该id截止本次交易所登陆过的不同的device、ip、city数
def get_till_now_device_ip_city_sum_num():
    global trade_df, merged_login_df, outputdir
    start_t = time.time()
    trade_df_new = trade_df.copy()
    trade_df_new = trade_df_new.sort_values(by='time')
#    trade_df_new = trade_df_new.iloc[:100]
    device_num_list, ip_num_list, city_num_list = [], [], []
    count = 0
    print('starting computing')
    start_t = time.time()
    for index, row in trade_df_new.iterrows():
        print('check_different_device_ip_city compute ', count)
        count += 1
        trade_time = row['time']
        user_id = row['id']
        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        if login_sub_df.shape[0]>1:
            login_device_num = len(set(login_sub_df['device'].values()))
            login_ip_num = len(set(login_sub_df['ip'].values()))
            login_city_num = len(set(login_sub_df['city'].values()))
        else:
            login_device_num, login_ip_num, login_city_num = 0, 0, 0
        device_num_list.append(login_device_num)
        ip_num_list.append(login_ip_num)
        city_num_list.append(login_city_num)

    trade_df_new['till_now_login_device_num'] = np.array(device_num_list)
    trade_df_new['till_now_login_ip_num'] = np.array(ip_num_list)
    trade_df_new['till_now_login_city_num'] = np.array(city_num_list)
    print('cost time is ', end_t-start_t)

    trade_df_new.to_csv(outputdir + 'till_now_device_ip_city_sum_num.csv')


############################################################################################################
###------------------------------------------------------------------------------------------------------###
###------------------------------------------下面为穿越特征 ----------------------------------------------###
###------------------------------------------------------------------------------------------------------###
############################################################################################################



#------------------------------------------------------------------------------------#
#得到该id所登陆过的不同的device、ip、city数
def get_device_ip_city_sum_num():
    global trade_df, merged_login_df, outputdir
    start_t = time.time()
    trade_df_new = trade_df.copy()
    device_count_s = merged_login_df.groupby(['id', 'device']).size().to_frame().reset_index().groupby('id').size()
    trade_df_new['device_sum_num'] = device_count_s[trade_df_new['id']].values
    ip_count_s = merged_login_df.groupby(['id', 'ip']).size().to_frame().reset_index().groupby('id').size()
    trade_df_new['ip_sum_num'] = ip_count_s[trade_df_new['id']].values
    city_count_s = merged_login_df.groupby(['id', 'city']).size().to_frame().reset_index().groupby('id').size()
    trade_df_new['city_sum_num'] = city_count_s[trade_df_new['id']].values
    end_t = time.time()
    print('cost time is ', end_t-start_t)

    trade_df_new[['id', 'device_sum_num', 'ip_sum_num', 'city_sum_num']].to_csv(
        outputdir + 'till_now_device_ip_city_sum_num.csv')
    
    
    
if __name__=='__main__':
    a = np.timedelta64(1, 'Y')
    b_t = np.datetime64('2011-06-15 12:23:00') - np.timedelta64(12, 'D')
    print('b_t is ', b_t)
    
#    print(sys.version)
#    sss = pd.Series({'a': 3, 'b': 8, 'c': 5, 'd': 2})
#    sss.plot(kind='bar')
    
#    df1 = pd.DataFrame({'name': ['a', 'a', 'a', 'c', 'd', 'b', 'd'],
#                     'value':[12, 23, 33, 44, 55, 66, 77],
#                     'ttt':[111, 222, 333, 444, 555, 666, 777]})
#    df2 = pd.DataFrame({'name': ['d', 'b', 'a', 'c', 'd', 'a', 'c'],
#                     'value':[121, 231, 331, 441, 551, 661, 771]})
#    df_merged = pd.melt(df1)
#    df_merged = pd.concat([df1, df2]).reset_index()
#    print(df_merged)
    sys.exit(0)
    
#    trade_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade.csv', 
#                           index_col='rowkey', dtype={'id': np.str})
#    login_df = pd.read_csv('./data/Risk_Detection_Qualification/t_login.csv', 
#                           index_col='log_id', 
#                           dtype={'id': np.str, 'timestamp': np.str},
#                           parse_dates=['time'], date_parser=dateparse)

    start_t = time.time()
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    
    outputdir = './features/train/'
    trade_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade.csv', 
                           index_col='rowkey', parse_dates=['time'],
                           date_parser=dateparse)
    
    login_df = pd.read_csv('./data/Risk_Detection_Qualification/t_login.csv', 
                           index_col='log_id', parse_dates=['time'], 
                           date_parser=dateparse)
    login_test_df = pd.read_csv('./data/Risk_Detection_Qualification/t_login_test.csv', 
                                index_col='log_id', parse_dates=['time'],
                                date_parser=dateparse)
    merged_login_df = login_df.append(login_test_df)

    removed_duplicate_merged_login_df = pd.read_csv('./data/removed_duplicate.csv', 
                                index_col='log_id', parse_dates=['time'],
                                date_parser=dateparse)
    
#    check_835072_device()
#    check_different_device_ip_city()
    get_device_ip_city_sum_num()
    
#    remove_duplicate_login_records(merged_login_df)
#    get_sameday_sameid_different_trade_risk(trade_df)
    
#    get_neareast_N_change_city_min_elaspe()
#    get_last_login_device_ip_city_elapse()
#    get_whether_today_first_trade_login()
    
#    get_last_login_trade_time_elapse()
#    get_whether_this_trade_same_device_ip_city()
#    get_last3_login_info()
#    get_before_trade_trade_login_num()
#    get_from_2015_1_1_minutes_num()
#    get_from_last_login_trade_num()
#    get_whether_between_1_and_7_am()
    
    
#    df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [2, 4, 5, np.nan], 
#                       'C': [100, 200, 300, np.nan]})
#    df.to_csv('test_output111.csv')

#    outputdir = './features/test/'    
#    trade_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade_test.csv', 
#                           index_col='rowkey', parse_dates=['time'],
#                           date_parser=dateparse)
    
#    get_neareast_N_change_city_min_elaspe() 
#    get_last_login_device_ip_city_elapse() 
#    get_whether_today_first_trade_login() 
#    get_last_login_trade_time_elapse() 
#    get_whether_this_trade_same_device_ip_city() 
#    get_last3_login_info() #
#    get_before_trade_trade_login_num()
#    get_from_2015_1_1_minutes_num()
#    get_from_last_login_trade_num()
#    get_whether_between_1_and_7_am()
    
    end_t = time.time()
    print('total running cost time: ', end_t-start_t)
    
#    df = pd.read_csv('./features/test/last3_login_info.csv')
#    print('df is ', df.count())
#    result_df = pd.get_dummies(df['result1'])
#    print(result_df.head(20))
    
    
    
    
    
    
# merged_login_df['log_from'].unique().size
    
    
    
    
    
    
    
    
    
    
    
    
    