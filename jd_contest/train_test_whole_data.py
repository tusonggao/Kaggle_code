import numpy as np
np.random.seed(2017)

import sys
import time
from datetime import datetime
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle as skl_shuffle

import xgboost as xgb
from xgboost.sklearn import XGBClassifier


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    

def store_feature_importances(model, feature_names, name_affix=None):
    array = model.feature_importances_
    importance_df = pd.DataFrame({'feature_name': feature_names, 
                                  'importance_val': list(array)})
    importance_df.set_index(['feature_name'], inplace=True)
    importance_df.sort_values(by='importance_val', ascending=False,
                              inplace=True)
    if name_affix is None:
        name_affix = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
    importance_df.to_csv('./model_dumps/feature_importances_' + name_affix + '.csv')


def jd_score(y_true, y_predicted, beta=0.1):
    precision = metrics.precision_score(y_true, y_predicted)
    recall = metrics.recall_score(y_true, y_predicted)
    accuracy = metrics.accuracy_score(y_true, y_predicted)
    print('precision, recall, accuracy is ', precision, recall, accuracy)
    score = (1 + beta**2)*(precision*recall)/(beta**2*precision + recall)
    return score
    
def jd_score111(precision, recall, beta=0.1):
    score = (1 + beta**2)*(precision*recall)/(beta**2*precision + recall)
    return score

def train_test_split_new(X_train, y_train): # 按照时间划分 1-5月作为训练集 6月数据作为测试集
    merged_df = pd.concat([X_train, y_train], axis=1)
    merged_df.sort_values(by='time', inplace=True)
    train_part = merged_df[merged_df.time<np.datetime64('2015-06-01 00:00:00')]
    test_part = merged_df[merged_df.time>=np.datetime64('2015-06-01 00:00:00')]
    print('len of train_part is ', len(train_part))
    print('len of test_part is ', len(test_part))
    
    return (train_part.iloc[:, :-1], test_part.iloc[:, :-1],
            train_part.iloc[:, -1], test_part.iloc[:, -1])
 
    
def inblance_preprocessing(data_df, label_df):
    data_df = pd.concat([data_df, label_df], axis=1)
    positive_instances = data_df[data_df['is_risk']==1]
    negative_instances = data_df[data_df['is_risk']==0]

    print('positive_instances negative_instances len is ', 
          len(positive_instances), len(negative_instances))
    
    if len(positive_instances) > len(negative_instances):
        n = int(len(positive_instances)/len(negative_instances))
    else:
        n = int(len(negative_instances)/len(positive_instances))
    n = max(n, 1)
        
    all_instances =  negative_instances
    for _ in range(n):
        all_instances = all_instances.append(positive_instances)
    print('all_instances len is ', len(all_instances),
          'shape is ', all_instances.shape)
    all_instances = skl_shuffle(all_instances)

    return all_instances.iloc[:, :-1], all_instances.iloc[:, -1]

def training_with_gbdt(max_depth, learning_rate, 
                       n_estimators=600, sample_weight=None):
    global X_train, y_train, X_test, real_test_df
    
    print('in training_with_gbdt, max_depth={}, learning_rate={} n_estimators={}'.format(
          max_depth, learning_rate, n_estimators))
    gbdt = GradientBoostingClassifier(n_estimators=n_estimators, 
                                      max_depth=max_depth, 
                                      learning_rate=learning_rate,
                                      random_state=42,
                                      verbose=1)
    gbdt.fit(X_train, y_train, sample_weight=sample_weight)
#    gbdt.fit(X_train.iloc[:1000], y_train.iloc[:1000], sample_weight=sample_weight[:1000])
    
    print("accuracy on training set:", gbdt.score(X_train, y_train))
    outcome = gbdt.predict(X_test)
    score = jd_score(y_test, outcome)
    print('in validation test get score ', score)
    
    time_str = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
    name_affix = ('gbdt_whole_' + time_str + '_' + str(round(score, 5)) + '_depth_' + 
                  str(max_depth) + '_learningrate_' + str(learning_rate) + 
                  '_n_estimators_' + str(n_estimators))
                  
    store_feature_importances(gbdt, list(X_train.columns), name_affix)
    with open('./model_dumps/gbdt_' + name_affix + '.pkl', 'wb') as f:
        pickle.dump(gbdt, f)
    real_predicted_outcome = gbdt.predict(real_test_df)
    outcome_df = real_test_df.copy()
    outcome_df['is_risk'] = real_predicted_outcome
    outcome_df['is_risk'].to_csv('./data/submission/submission_'+ name_affix + '.csv')
    

def training_with_rf(n_estimators=3000, min_samples_leaf=3, sample_weight=None):
    global X_train, y_train, X_test, real_test_df
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                n_jobs=1,
                                random_state=42,
                                min_samples_leaf=min_samples_leaf,
                                oob_score=True,
                                verbose=1
                                )
    print('in training_with_rf, n_estimators={}, min_samples_leaf={}'.format(
          n_estimators, min_samples_leaf))
    rf.fit(X_train, y_train, sample_weight=sample_weight)
#    rf.fit(X_train.iloc[:1000], y_train.iloc[:1000])
    outcome = rf.predict(X_test)
    score = jd_score(y_test, outcome)
    print('in random forest validation test get score ', score)
    
    time_str = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
    name_affix = ('rf_whole_' + time_str + '_' + str(round(score, 5)) + '_n_estimators_' + 
                  str(n_estimators) + '_min_samples_leaf_' + str(min_samples_leaf))
    store_feature_importances(rf, list(X_train.columns), name_affix)
    with open('./model_dumps/rf_' + name_affix + '.pkl', 'wb') as f:
        pickle.dump(rf, f)
    real_predicted_outcome = rf.predict(real_test_df)
    outcome_df = real_test_df.copy()
    outcome_df['is_risk'] = real_predicted_outcome
    outcome_df['is_risk'].to_csv('./data/submission/submission_'+ name_affix + '.csv')
    
    

if __name__=='__main__':

    start_t = time.time()
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')    
    train_df = pd.read_csv('./data/train_data.csv', index_col='rowkey',
                           parse_dates=['time'], date_parser=dateparse)
    print('len of train_df is ', len(train_df))

    _, X_test, _, y_test = train_test_split_new(
                         train_df.iloc[:, :-1], train_df.iloc[:, -1])
    
    X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    
    weight_arr = np.where(y_train==1, 1, 0.0283)
    print(weight_arr)
    
    
    X_train.drop('time', axis=1, inplace=True)
#    X_train.drop('id', axis=1, inplace=True)
    X_test.drop('time', axis=1, inplace=True)
#    X_test.drop('id', axis=1, inplace=True)
    
    real_test_df = pd.read_csv('./data/test_data.csv', index_col='rowkey',
                               parse_dates=['time'], date_parser=dateparse)
    real_test_df.drop('time', axis=1, inplace=True)
#    real_test_df.drop('id', axis=1, inplace=True)

    training_with_rf(sample_weight=weight_arr)
    
#    max_depth_list = [10]
#    learning_rate_list = [0.11]
#    for depth in max_depth_list:
#        for rate in learning_rate_list:
#            print('preprocessing depth {} rate {}'.format(depth, rate))
#            training_with_gbdt(depth, rate, 
#                               n_estimators=600, 
#                               sample_weight=weight_arr)

    end_t = time.time()
    print('total cost time is ', end_t-start_t)
    
    
    
    
    
