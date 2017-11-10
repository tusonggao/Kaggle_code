import numpy as np
np.random.seed(2017)
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split


def jd_score(y_true, y_predicted, beta=0.1):
    precision = metrics.precision_score(y_true, y_predicted)
    recall = metrics.recall_score(y_true, y_predicted)
    print('precision, recall is ', precision, recall)
    score = (1 + beta**2)*(precision*recall)/(beta**2*precision + recall)
    return score

    
def load_data():
    trade_dtype = {'id': np.str, 'time': np.datetime64}
    trade_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade.csv', 
                           index_col='rowkey',
                           dtype=trade_dtype)
    login_dtype = {'id': np.str, 'timestamp': np.int32}
    login_df = pd.read_csv('./data/Risk_Detection_Qualification/t_login.csv', 
                           index_col='log_id', 
                           dtype=login_dtype)
    X_train, X_test, y_train, y_test = train_test_split(trade_df)

def random_select(test_data):
    trade_dtype = {'id': np.str, 'time': np.datetime64}
    trade_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade.csv', 
                           index_col='rowkey',
                           dtype=trade_dtype)
    login_dtype = {'id': np.str, 'timestamp': np.int32}
    login_df = pd.read_csv('./data/Risk_Detection_Qualification/t_login.csv', 
                           index_col='log_id', 
                           dtype=login_dtype)
    X_train, X_test, y_train, y_test = train_test_split(trade_df)

    
if __name__=='__main__':
    print(jd_score([1, 0, 1, 1, 0], [1, 0, 1, 0, 1]))
    