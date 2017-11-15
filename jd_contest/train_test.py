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

def training_with_gbdt(max_depth, learning_rate):
    global X_train, y_train, X_test
    gbdt = GradientBoostingClassifier(n_estimators=100, 
                                      max_depth=max_depth, 
                                      learning_rate=learning_rate,
                                      random_state=42,
                                      verbose=1)
#    gbdt.fit(X_train, y_train)
    gbdt.fit(X_train.iloc[:1000], y_train.iloc[:1000])
    
#    gbdt = GradientBoostingClassifier(n_estimators=100, max_depth=6, 
#                                     learning_rate=0.05,
#                                     random_state=42,
#                                     verbose=1)        
#    gbdt.fit(X_train.iloc[:1000], y_train.iloc[:1000])
    
    print("accuracy on training set:", gbdt.score(X_train, y_train))
    outcome = gbdt.predict(X_test)
    score = jd_score(y_test, outcome)
    print('in validation test get score ', score)

#    pickle_in = open('./model_dumps/gbdt_2017-11-15_03_25_49.pkl', 'rb')
#    gbdt = pickle.load(pickle_in)
    
#    print("gbt accuracy on training set:", gbdt.score(X_train, y_train))
#    training_jd_score = jd_score(y_train, gbdt.predict(X_train))
#    print('training_jd_score is ', training_jd_score)
#    test_jd_score = jd_score(y_test, gbdt.predict(X_test))
#    print('test_jd_score is ', test_jd_score)
    
    
    real_test_df = pd.read_csv('./data/test_data.csv', index_col='rowkey',
                               parse_dates=['time'], date_parser=dateparse)
    real_test_df.drop('time', axis=1, inplace=True)
    real_test_df.drop('id', axis=1, inplace=True)
    
    time_str = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
    name_affix = (time_str + '_' + str(round(score, 5)) + '_depth_' + 
                  str(max_depth) + '_learningrate_' + str(learning_rate))
    store_feature_importances(gbdt, list(X_train.columns), name_affix)
    with open('./model_dumps/gbdt_' + name_affix + '.pkl', 'wb') as f:
        pickle.dump(gbdt, f)
    real_predicted_outcome = gbdt.predict(real_test_df)
    real_test_df['is_risk'] = real_predicted_outcome
    real_test_df['is_risk'].to_csv('./data/submission/submission_'+ name_affix + '.csv')
    
    
    

if __name__=='__main__':
#    print(time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime()))
#    print(jd_score111(0.45, 0.090))
#    sys.exit(0)    

    start_t = time.time()
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')    
    train_df = pd.read_csv('./data/train_data.csv', index_col='rowkey',
                           parse_dates=['time'], date_parser=dateparse)
    print('len of train_df is ', len(train_df))
    X_train, X_test, y_train, y_test = train_test_split_new(
                         train_df.iloc[:, :-1], train_df.iloc[:, -1])
    
    X_train.drop('time', axis=1, inplace=True)
    X_train.drop('id', axis=1, inplace=True)
    X_test.drop('time', axis=1, inplace=True)
    X_test.drop('id', axis=1, inplace=True)
    
    print('111 X_train X_test', X_train.shape, X_test.shape, 
          y_train.shape, y_test.shape)
    
    X_train, y_train = inblance_preprocessing(X_train, y_train)
    print('222 X_train y_train', X_train.shape, y_train.shape)
    
#    gbdt = GradientBoostingClassifier(n_estimators=1000, max_depth=13, 
#                                     learning_rate=0.05,
#                                     random_state=42,
#                                     verbose=1)
    

#    max_depth_list = [7, 9, 11, 13]
#    learning_rate_list = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13]
    max_depth_list = [7]
    learning_rate_list = [0.01]
    for depth in max_depth_list:
        for rate in learning_rate_list:
            print('preprocessing depth {} rate {}'.format(depth, rate))
            training_with_gbdt(depth, rate)

#    gbdt.fit(X_train, y_train)
    
#    gbdt = GradientBoostingClassifier(n_estimators=100, max_depth=6, 
#                                     learning_rate=0.05,
#                                     random_state=42,
#                                     verbose=1)        
#    gbdt.fit(X_train.iloc[:1000], y_train.iloc[:1000])
    
#    print("accuracy on training set:", gbdt.score(X_train, y_train))
#    outcome = gbdt.predict(X_test)
#    score = jd_score(y_test, outcome)
#    print('in validation test get score ', score)

#    pickle_in = open('./model_dumps/gbdt_2017-11-15_03_25_49.pkl', 'rb')
#    gbdt = pickle.load(pickle_in)
    
#    print("gbt accuracy on training set:", gbdt.score(X_train, y_train))
#    training_jd_score = jd_score(y_train, gbdt.predict(X_train))
#    print('training_jd_score is ', training_jd_score)
#    test_jd_score = jd_score(y_test, gbdt.predict(X_test))
#    print('test_jd_score is ', test_jd_score)
    
    
#    real_test_df = pd.read_csv('./data/test_data.csv', index_col='rowkey',
#                               parse_dates=['time'], date_parser=dateparse)
#    real_test_df.drop('time', axis=1, inplace=True)
    
#    real_test_df.drop('id', axis=1, inplace=True)
    
#    time_str = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
#    name_affix = time_str + '_' + str(round(score, 5))
#    store_feature_importances(gbdt, list(X_train.columns), name_affix)
#    with open('./model_dumps/gbdt_' + name_affix + '.pkl', 'wb') as f:
#        pickle.dump(gbdt, f)
#    real_predicted_outcome = gbdt.predict(real_test_df)
#    real_test_df['is_risk'] = real_predicted_outcome
#    real_test_df['is_risk'].to_csv('./data/submission/submission_'+ name_affix + '.csv')
    

    end_t = time.time()
    print('total cost time is ', end_t-start_t)
    
    
    
    
    
    
    
#    svr grid_search.best_params_ is  {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 5000}
#svr grid_search.best_score_ is  0.848484848485
#best score is  0.997755331089
#time cost is  6576.964999914169

#    param_grid = {'n_estimators': [300],
#                  'learning_rate': [0.001, 0.001, 0.1],
#                  'max_depth': [2, 4, 6, 8, 10, None]}
#                  
#    grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), 
#                               param_grid, cv=5)
#    grid_search.fit(X_train, y_train)
#    test_score = grid_search.score(X_train, y_train)
#    outcome = grid_search.predict(X_test)
#    gbr_prob = grid_search.predict_proba(X_test)
#    
#    print('svr grid_search.best_params_ is ', grid_search.best_params_)
#    print('svr grid_search.best_score_ is ', grid_search.best_score_)
#    print('best score is ', test_score)
    

#    scores = cross_val_score(gbr, X_train, y_train, cv=5)
#    print('scores mean is ', scores.mean())
        
#    gbr_prob = gbr.predict_proba(X_test)
#    print('gbr_prob is ', gbr_prob)
#    print('shape of gbr_prob is ', gbr_prob.shape)


    
    
#    gbr_prob_frame = pd.DataFrame({'0': gbr_prob[:, 0], 
#                                   '1':gbr_prob[:, 1],
#                                   'outcome': outcome},
#                                   index=X_test.index.values)
#    gbr_prob_frame.to_csv('./gbr_prob_frame.csv', index_label='PassengerId')