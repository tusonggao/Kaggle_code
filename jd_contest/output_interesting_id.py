import numpy as np
np.random.seed(2017)
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
    min_elapse = np.nan
    for i in range(len(login_df)):
        ip, city, time = login_df.iloc[i, ['ip', 'city', 'time']]
    for index, row in trade_df.iterrows():
    


def get_neareast_N_login(N=20):
    global trade_df, merged_login_df
    trade_df_new = trade_df.copy()
    change_city_min_elapse = []
    count = 0
    for index, row in trade_df.iterrows():
        count += 1
        trade_time = row['row']
        user_id = row['id']
        login_sub_df = merged_login_df[merged_login_df.id==user_id].sort_values(by='time')
        login_sub_df = login_sub_df[login_sub_df.time<=trade_time]
        login_sub_df = login_sub_df.iloc[-N:, :]
        min_elapses.append(get_change_city_min_elapse(login_sub_df))

        print(trade_time)
        if count >= 1:
            break
    trade_df_new[]
    
if __name__=='__main__':
#    trade_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade.csv', 
#                           index_col='rowkey', dtype={'id': np.str})
#    trade_test_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade_test.csv', 
#                                index_col='rowkey', dtype={'id': np.str})
#    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
#    login_df = pd.read_csv('./data/Risk_Detection_Qualification/t_login.csv', 
#                           index_col='log_id', 
#                           dtype={'id': np.str, 'timestamp': np.str},
#                           parse_dates=['time'], date_parser=dateparse)
#    login_test_df = pd.read_csv('./data/Risk_Detection_Qualification/t_login_test.csv', 
#                                index_col='log_id', 
#                                dtype={'id': np.str, 'timestamp': np.str},
#                                parse_dates=['time'], date_parser=dateparse)
#    merged_login_df = login_df.append(login_test_df)
#    
#    get_neareast_N_login()
    
    xs = np.array([1, 2, 3, 4, 5])
    city_shifted = np.array([True, False, True, False, True])
    print(xs[-2:])
    
    
#    xs_shifted = np.roll(xs, 2)    
    print('xs_shifted is ', xs[~city_shifted])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    