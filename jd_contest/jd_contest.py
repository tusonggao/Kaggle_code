import numpy as np
np.random.seed(2017)
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split

def jd_score(y_true, y_predicted, beta=0.1):
    precision = metrics.precision_score(y_true, y_predicted)
    recall = metrics.recall_score(y_true, y_predicted)
    accuracy = metrics.accuracy_score(y_true, y_predicted)
    print('precision, recall, accuracy is ', precision, recall, accuracy)
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
#    X_train, X_test, y_train, y_test = train_test_split(trade_df)

def random_select():
#    trade_dtype = {'id': np.str, 'time': np.datetime64}
    trade_dtype = {'id': np.str}
    train_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade.csv', 
                           dtype=trade_dtype)
    
    X_train_original = train_df['rowkey']
    y_train_original = train_df['is_risk']
    X_train, X_test, y_train, y_test = train_test_split(X_train_original, 
                                          y_train_original,
                                          test_size=0.25,
                                          random_state=42)
    return X_train, X_test, y_train, y_test

def random_train(y_test, ratio=0.7):
    random_arr = np.random.random(len(y_test))
    y_hat = np.where(random_arr<ratio, 0, 1)
    return y_hat
    
if __name__=='__main__':
    print(jd_score([1, 0, 1, 1, 0], [1, 0, 1, 0, 1]))
    
    X_train, X_test, y_train, y_test = random_select()
#    print('y_train is ', len(y_train))
#    print('y_test is ', len(y_test))
    y_hat = random_train(y_test, ratio=0.001)   
    
    print(jd_score(y_test, y_hat))
#    print(jd_score(y_test, y_test))
    
    print('len of y_test is ', len(y_test), np.unique(y_test))
    print('len of y_hat is ', len(y_hat), np.unique(y_hat))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    