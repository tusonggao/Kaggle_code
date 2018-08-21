import pickle
import gc
import math
import time
import os
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline

def convert_2_md5(value):
    return hashlib.md5(str(value).encode('utf-8')).hexdigest()

def write_to_log(*param):
    param_list = [str(s) for s in param]
    log = ' '.join(param_list)
    with open('C:/D_Disk/data_competition/model_stacking'
              '/outcome/log_file.txt', 'a') as file:
        file.write(log+'\n')
        file.flush()  #立即写入磁盘
        os.fsync(file)  #立即写入磁盘

SEED = 222

train_df = pd.read_csv('C:/D_Disk/data_competition/model_stacking/input_train.csv', index_col=0)
test_df = pd.read_csv('C:/D_Disk/data_competition/model_stacking/input_test.csv', index_col=0)

public_test_df = pd.read_csv('C:/D_Disk/data_competition/model_stacking/input_test_public.csv', index_col=0)
private_test_df = pd.read_csv('C:/D_Disk/data_competition/model_stacking/input_test_private.csv', index_col=0)


df = pd.read_csv('C:/D_Disk/data_competition/model_stacking/input.csv')
df.loc[df['cand_pty_affiliation']=='DEM', 'cand_pty_affiliation'] = 1
df.loc[df['cand_pty_affiliation']=='REP', 'cand_pty_affiliation'] = 0
test_y = df.loc[test_df.index, 'cand_pty_affiliation'].values


train_y = train_df['cand_pty_affiliation']
train_df.drop(['cand_pty_affiliation'], axis=1, inplace=True)


merged_df = train_df.append(test_df)
print('before get_dummies merged_df.shape is ', merged_df.shape)
merged_df = pd.get_dummies(merged_df, sparse=True)

train_X = merged_df[:len(train_df)]
test_X = merged_df[len(train_df):]
print('after get_dummies merged_df.shape is ', merged_df.shape)

def test_param_lgbm(lgbm_param, folds=5):
    start_t = time.time()
    
    auc_score_list = []
    y_pred_result = 0
    for fold in range(folds):
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
                train_X, train_y, test_size=0.2, random_state=fold)
        lgbm_param['random_state'] = fold

        lgbm = lgb.LGBMClassifier(**lgbm_param)
        lgbm.fit(X_train_new, y_train_new, eval_set=[(X_val_new, y_val_new)], 
                eval_metric='auc', verbose=400, early_stopping_rounds=200)
		
        y_predictions_val = lgbm.predict_proba(X_val_new)[:, 1]
        val_auc_score = roc_auc_score(y_val_new, y_predictions_val)
        auc_score_list.append(val_auc_score)
        print('round: ', fold, ' rmse for validation data is ', val_auc_score)
        
        y_pred = lgbm.predict_proba(test_X)[:, 1]
        y_pred_result += y_pred
    
    auc_score_avg = round(np.array(auc_score_list).mean(), 5)
    print('auc_score_list is ', auc_score_list, 'auc_score_avg is ', auc_score_avg)
    
    store_path = 'C:/D_Disk/data_competition/model_stacking/outcome/'
    param_md5_str = convert_2_md5(lgbm_param)
    full_file_name = ('_'.join(['submission_lgbm',
                               str(auc_score_avg), 
                               param_md5_str])
                       + '.csv')
    y_pred_result /= folds
    test_df.loc[test_X.index, 'cand_pty_affiliation'] = y_pred_result
    test_df['cand_pty_affiliation'].to_csv(store_path + full_file_name, 
           header=['cand_pty_affiliation'])
    
    public_test_y_pred = test_df.loc[public_test_df.index, 'cand_pty_affiliation']
    public_test_y_true = public_test_df['cand_pty_affiliation']
    
    private_test_y_pred = test_df.loc[private_test_df.index, 'cand_pty_affiliation']
    private_test_y_true = private_test_df['cand_pty_affiliation']
    
    print('public_auc is ', roc_auc_score(public_test_y_true, public_test_y_pred),
          'private_auc is ', roc_auc_score(private_test_y_true, private_test_y_pred),
          'get partial predict outcome cost time: ', time.time()-start_t)
    
    write_to_log('-'*25, ' lgbm  md5 value: ', param_md5_str, '-'*25)
    write_to_log('param: ', lgbm_param)
    write_to_log('valid auc avg: ', auc_score_avg)
    write_to_log('-'*80+'\n')

    
def test_param_gbdt_whole(param):
    print('in test_param_gbdt_whole')
    start_t = time.time()
    
    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
                train_X, train_y, test_size=0.2, random_state=SEED)

    param['validation_fraction'] = 0.2
    param['n_iter_no_change'] = 300
    param['verbose'] = 1
    n_estimators_now = 0
    n_estimators_max = 8000
    n_estimators_step = 200
    
    auc_newest = 0.0
    clf = GradientBoostingClassifier(**param)
    
    
#    while n_estimators_now < n_estimators_max:
#        n_estimators_now += n_estimators_step
##        param['n_estimators'] = n_estimators_now
#        start_t = time.time()
#        clf.set_params(n_estimators=n_estimators_now)
#        clf.fit(X_train_new, y_train_new)
#        print('training cost time: ', time.time()-start_t)
#        y_predictions_val = clf.predict_proba(X_val_new)[:, 1]
#        val_auc_score = roc_auc_score(y_val_new, y_predictions_val)
#        print('n_estimators_now: ', n_estimators_now, 
#              'newest auc is ', val_auc_score)
#        if auc_newest < val_auc_score:
#            auc_newest = val_auc_score
#        else:
#            print('auc score stop to increase in valid data set, exit the loop')
#            break
    
    clf.fit(train_X, train_y)
    
    print('clf.estimators_ is ', clf.estimators_)
    
#    param['warm_start'] = False
#    param['n_estimators'] = n_estimators_now
#    param['n_iter_no_change'] = None
#    param['verbose'] = 1
#    clf = GradientBoostingClassifier(**param)
#    print('training with whole data:')
#    clf.fit(train_X, train_y)
        
    y_pred_train = clf.predict_proba(train_X)[:, 1]
    auc_train = roc_auc_score(train_y, y_pred_train)
    print('auc_train is ', auc_train)
    
    y_pred = clf.predict_proba(test_X)[:, 1]
    test_df.loc[test_X.index, 'cand_pty_affiliation'] = y_pred
    
    public_test_y_pred = test_df.loc[public_test_df.index, 'cand_pty_affiliation']
    public_test_y_true = public_test_df['cand_pty_affiliation']
    
    private_test_y_pred = test_df.loc[private_test_df.index, 'cand_pty_affiliation']
    private_test_y_true = private_test_df['cand_pty_affiliation']
    
    print('public_auc is ', roc_auc_score(public_test_y_true, public_test_y_pred),
          'private_auc is ', roc_auc_score(private_test_y_true, private_test_y_pred),
          'get partial predict outcome cost time: ', time.time()-start_t)
    

def test_param_xgb(param, folds=5):
    start_t = time.time()
    
    auc_score_list = []
    y_pred_result = 0
    for fold in range(folds):
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
                train_X, train_y, test_size=0.2, random_state=fold)
        param['random_state'] = fold

        clf = xgb.XGBClassifier(**param)
        clf.fit(X_train_new, y_train_new, eval_set=[(X_val_new, y_val_new)], 
                eval_metric='auc', verbose=400, early_stopping_rounds=200)
		
        y_predictions_val = clf.predict_proba(X_val_new)[:, 1]
        val_auc_score = roc_auc_score(y_val_new, y_predictions_val)
        auc_score_list.append(val_auc_score)
        print('round: ', fold, ' rmse for validation data is ', val_auc_score)
        
        y_pred = clf.predict_proba(test_X)[:, 1]
        y_pred_result += y_pred
    
    auc_score_avg = round(np.array(auc_score_list).mean(), 5)
    print('auc_score_list is ', auc_score_list, 'auc_score_avg is ', auc_score_avg)
    
    store_path = 'C:/D_Disk/data_competition/model_stacking/outcome/'
    param_md5_str = convert_2_md5(param)
    full_file_name = ('_'.join(['submission_xgb',
                               str(auc_score_avg), 
                               param_md5_str])
                       + '.csv')
    y_pred_result /= folds
    test_df.loc[test_X.index, 'cand_pty_affiliation'] = y_pred_result
    test_df['cand_pty_affiliation'].to_csv(store_path + full_file_name, 
           header=['cand_pty_affiliation'])
    
    public_test_y_pred = test_df.loc[public_test_df.index, 'cand_pty_affiliation']
    public_test_y_true = public_test_df['cand_pty_affiliation']
    
    private_test_y_pred = test_df.loc[private_test_df.index, 'cand_pty_affiliation']
    private_test_y_true = private_test_df['cand_pty_affiliation']
    
    print('public_auc is ', roc_auc_score(public_test_y_true, public_test_y_pred),
          'private_auc is ', roc_auc_score(private_test_y_true, private_test_y_pred),
          'get partial predict outcome cost time: ', time.time()-start_t)
    
    write_to_log('-'*25, 'xgb model ', ' md5 value: ', param_md5_str, '-'*25)
    write_to_log('param: ', lgbm_param)
    write_to_log('valid auc avg: ', auc_score_avg)
    write_to_log('-'*80+'\n')

def test_param_xgboost_whole(param):
    start_t = time.time()
    
    clf = xgb.XGBClassifier(**param)
    clf.fit(train_X, train_y)
    
    y_pred_train = clf.predict_proba(train_X)[:, 1]
    auc_train = roc_auc_score(train_y, y_pred_train)
    print('auc_train is ', auc_train)
    
    y_pred = clf.predict_proba(test_X)[:, 1]
    test_df.loc[test_X.index, 'cand_pty_affiliation'] = y_pred
    
    public_test_y_pred = test_df.loc[public_test_df.index, 'cand_pty_affiliation']
    public_test_y_true = public_test_df['cand_pty_affiliation']
    
    private_test_y_pred = test_df.loc[private_test_df.index, 'cand_pty_affiliation']
    private_test_y_true = private_test_df['cand_pty_affiliation']
    
    print('public_auc is ', roc_auc_score(public_test_y_true, public_test_y_pred),
          'private_auc is ', roc_auc_score(private_test_y_true, private_test_y_pred),
          'get partial predict outcome cost time: ', time.time()-start_t)



def merge_outcome(outcome_f_n_1, outcome_f_n_2):
    start_t = time.time()
    
    print('hello world!')
    outcome_df1 = pd.read_csv('C:/D_Disk/data_competition/model_stacking/outcome/'+
                                 outcome_f_n_1, index_col=0)
    outcome_df2 = pd.read_csv('C:/D_Disk/data_competition/model_stacking/outcome/'+
                                 outcome_f_n_2, index_col=0)
    
    
    print('corr is ', 
          pd.DataFrame({"xgb": outcome_df1['cand_pty_affiliation'].values, 
                        "lgbm": outcome_df2['cand_pty_affiliation'].values}).corr())
    
    
    outcome_df1['cand_pty_affiliation'] += 2*outcome_df2['cand_pty_affiliation']
    outcome_df1['cand_pty_affiliation'] /= 3.0
    
    public_test_y_pred = outcome_df1.loc[public_test_df.index, 'cand_pty_affiliation']
    public_test_y_true = public_test_df['cand_pty_affiliation']
    
    private_test_y_pred = outcome_df1.loc[private_test_df.index, 'cand_pty_affiliation']
    private_test_y_true = private_test_df['cand_pty_affiliation']
    
    print('public_auc is ', roc_auc_score(public_test_y_true, public_test_y_pred),
          'private_auc is ', roc_auc_score(private_test_y_true, private_test_y_pred),
          'outcome cost time: ', time.time()-start_t)
    

def get_models():
    """Generate a library of base learners."""
    nb = GaussianNB()
    svc = SVC(C=100, probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
    lr = LogisticRegression(C=100, random_state=SEED)
    nn = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=1000, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=SEED)

    models = {'svm': svc,
              'knn': knn,
              'naive bayes': nb,
              'mlp-nn': nn,
              'random forest': rf,
              'gbm': gb,
              'logistic': lr,
              }

    return models


def train_predict(model_list):
    """Fit models in list on training set and return preds"""
    P = np.zeros((test_X.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        start_t = time.time()
        print("%s..." % name, end="   ", flush=False)
        m.fit(train_X, train_y)
        P.iloc[:, i] = m.predict_proba(test_X)[:, 1]
        cols.append(name)
        print('done, cost time', time.time()-start_t)

    P.columns = cols
    print("Done.\n")
    return P


def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        score = roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")

#param = {'n_estimators':8000, 'n_jobs':-1, 'learning_rate':0.015, 
#         'random_state':42, 'silent':-1, 'verbose':-1}

#param = {'n_estimators':8000, 'learning_rate':0.02, 'max_features':11,
#         'random_state':42, 'verbose':1}

param = {'n_estimators':8000, 'learning_rate':0.02, 'max_depth':5,
         'random_state':42, 'verbose':0}

#test_param_lgbm(param)
#test_param_lgbm_whole(param)

#test_param_xgb(param)

#test_param_xgboost_whole(param)


test_param_gbdt_whole(param)

#merge_outcome('submission_xgb_0.91326_0f8bab44a3fce00e3c9910186462d8a2.csv',
#              'submission_lgbm_0.91608_2ea925e4363c7f24373e0012bfcb336f.csv')
#
#models = get_models()
#P = train_predict(models)
#score_models(P, test_y)
#
#from mlens.visualization import corrmat
#
#corrmat(P.corr(), inflate=False)
#plt.show()


