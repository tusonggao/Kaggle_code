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
#        print('time time_next is ', time, time_next)
        if city!=city_next:
#            elapse = (time_next - time).astype('timedelta64[s]')
            elapse = (time_next - time)
            elapse = np.timedelta64(elapse, 's')
            elapse = elapse.astype('float')/60.0  # 计算间隔的分钟数，转换为float类型
            min_elapse = min(min_elapse, elapse)
        
    return min_elapse
    

def get_neareast_N_login(N=20):
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
#        print('login_sub_df is ', login_sub_df)
        elaspe_time = get_change_city_min_elapse(login_sub_df)
        min_elapses.append(elaspe_time)
#        print('user_id: {} trade_time: {} elaspe_time: {}'.format(user_id, trade_time, elaspe_time))
#        if count > 10:
#            break
    end_t = time.time()
    print('cost time is ', end_t-start_t)
    trade_df_new['change_city_min_elaspe'] = np.array(min_elapses)
    trade_df_new.to_csv('change_city_min_elaspe.csv')
    
    
    
    
    
    
if __name__=='__main__':
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

    
    get_neareast_N_login()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    