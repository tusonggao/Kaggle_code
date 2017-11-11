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
    
def jd_score111(beta=0.1):
    precision = 0.85
    recall = 0.9
    score = (1 + beta**2)*(precision*recall)/(beta**2*precision + recall)
    print('precision, recall, score is ', precision, recall, score)
    return score
    
def preprocess_data():
    trade_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade.csv', index_col='rowkey', dtype={'id': np.str})
    trade_test_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade_test.csv', index_col='rowkey', dtype={'id': np.str})
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    login_df = pd.read_csv('./data/Risk_Detection_Qualification/t_login.csv', index_col='log_id', 
                       dtype={'id': np.str, 'timestamp': np.str},
                       parse_dates=['time'], date_parser=dateparse)
    login_test_df = pd.read_csv('./data/Risk_Detection_Qualification/t_login_test.csv', index_col='log_id', 
                            dtype={'id': np.str, 'timestamp': np.str},
                            parse_dates=['time'], date_parser=dateparse)
    merged_login_df = login_df.append(login_test_df)
    
    X_train = trade_df[:, :-1]
    y_train = trade_df[:, -1]
    return X_train, y_train
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


def random_train(y_test, ratio=0.7):  # 随机判断
    ratio = len(y_test[y_test==1])/len(y_test) # 计算1所占比例
    print('len1 len2 is ', len(y_test[y_test==1]), len(y_test))
    print('computed ratio is ', ratio)
    random_arr = np.random.random(len(y_test))
    y_hat = np.where(random_arr<ratio, 0, 1)
    return y_hat

def decide_by_risk_id():  # 根据是否在trade_test中是否为risk id来判断，如果是的话，则其交易全部设置为risk
    trade_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade.csv', 
                           index_col='rowkey', dtype={'id': np.str})
    trade_test_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade_test.csv', 
                                index_col='rowkey', dtype={'id': np.str})    
    risk_id = trade_df[trade_df.is_risk==1]['id'].unique()
    trade_test_df['is_risk'] = 0
    trade_test_df['is_risk'][trade_test_df.id.isin(risk_id)] = 1
    results_df = trade_test_df['is_risk']
    results_df.to_csv('results_by_id.csv')
    
def jd_score_by_id():  # 只有 0.606523111616
    trade_df = pd.read_csv('./data/Risk_Detection_Qualification/t_trade.csv', 
                           index_col='rowkey', dtype={'id': np.str})
    trade_df_results = trade_df.copy()
    trade_df_results['is_risk'] = 0
    risk_id = trade_df[trade_df.is_risk==1]['id'].unique()
    trade_df_results['is_risk'][trade_df_results['id'].isin(risk_id)] = 1
    print('jd_score is ', jd_score(trade_df['is_risk'], trade_df_results['is_risk']))
    
    

if __name__=='__main__':
#    print(preprocess_data())
    
#    decide_by_risk_id()
    
#    jd_score_by_id()
    jd_score111()
#    check_data()
    
#    print(jd_score([1, 0, 1, 1, 0], [1, 0, 1, 0, 1]))    
#    X_train, X_test, y_train, y_test = random_select()
#    y_hat = random_train(y_test, ratio=0.001)       
#    print(jd_score(y_test, y_hat))    
#    print('len of y_test is ', len(y_test), np.unique(y_test))
#    print('len of y_hat is ', len(y_hat), np.unique(y_hat))

#    train_df = pd.read_csv('smalldata.csv')
#    print(train_df.count())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    