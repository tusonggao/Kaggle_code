import numpy as np
np.random.seed(2017)
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split

def get_one():
    global trade_df, merged_login_df
    risk_id = trade_df[trade_df.is_risk==1]['id'].unique()
    risk_login_df = merged_login_df[merged_login_df.id.isin(risk_id)]
    count_df = risk_login_df.groupby(risk_login_df['id']).count()
    print('count_df is ', count_df)

    
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
    
    get_one()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    