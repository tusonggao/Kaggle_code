import numpy as np
np.random.seed(2017)

import time
import pandas as pd
import matplotlib.pyplot as plt

def generate_training_dataframe():
    global trade_df
    
def generate_test_dataframe():
    global trade_test_df
    pass

def expand_df(merged_df):
    pass
    return merged_df
    
    
if __name__=='__main__':
    start_t = time.time()
    trade_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade.csv', 
                           index_col='rowkey', dtype={'id': np.str})
    trade_test_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade_test.csv', 
                                index_col='rowkey', dtype={'id': np.str})
    end_t = time.time()    
    print('load cost time: ', end_t-start_t)
    
    train_num = len(trade_df)
    train_y = trade_df['is_risk']
    
    training_df = generate_training_dataframe()
    test_df = generate_test_dataframe()
    
    merged_df = training_df.append(test_df)
    merged_df = expand_df(merged_df)
    train_df = pd.concat(merged_df.iloc[:train_num], train_y)
    test_df = merged_df.iloc[train_num:]

    train_df.to_csv('./data/train_data.csv')
    test_df.to_csv('./data/test_data.csv')

    
#    df = pd.read_csv('./features/test/last3_login_info.csv')
#    print('df is ', df.count())
#    result_df = pd.get_dummies(df['result1'])
#    print(result_df.head(20))
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    