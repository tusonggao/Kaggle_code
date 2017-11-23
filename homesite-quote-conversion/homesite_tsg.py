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



    
#def get_AB_test_score(y_test, y_predicted):
#    index_arr = np.random.shuffle(np.arange(0, len(y_test)))
#    index_A, index_B = np.split(index_arr, 2)
#    return index_A, index_B
#    score = jd_score(y_test, y_predicted)
#    score_a = jd_score(y_test[index_A], y_predicted[index_A])
#    score_b = jd_score(y_test[index_B], y_predicted[index_B])
#    return score, score_a, score_b
    
def get_AB_test_score(y_test, y_predicted):
    index_arr = np.arange(0, len(y_test))
    np.random.shuffle(index_arr)
    index_A, index_B = np.split(index_arr, 2)
    print('len A B is', len(index_A), len(index_B))
    
    y_test.index = np.arange(0, len(y_test))
    score = jd_score(y_test, y_predicted)
    score_a = jd_score(y_test[index_A], y_predicted[index_A])
    score_b = jd_score(y_test[index_B], y_predicted[index_B])
    print('jd score for test score: {} score_a: {} score_b: {}'.format(
          score, score_a, score_b))
    return score, score_a, score_b
    

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

def training_with_xgboost(max_depth, learning_rate, n_estimators=600,
                          subsample=1.0, negative_weight_ratio=1.0):
    start_t = time.time()
    print('in training_with_gbdt, max_depth={}, learning_rate={} '
          'n_estimators={} subsample={}'.format(
          max_depth, learning_rate, n_estimators, subsample))
    global X_train, y_train, X_test, y_test, real_test_df
    
    positive_instances = y_train[y_train==1]
    negative_instances = y_train[y_train==0]
    negative_weight = (negative_weight_ratio*len(positive_instances)/
                       len(negative_instances))
    print('negative_weight is ', negative_weight)
#    sample_weight = np.where(y_train==1, 1, negative_weight)
    sample_weight = np.where(y_train==1, 1, 1)
    
    xgbc = XGBClassifier(
        silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        #nthread=4,# cpu 线程数 默认最大
        learning_rate=learning_rate, # 如同学习率
        min_child_weight=1,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        max_depth=max_depth, # 构建树的深度，越大越容易过拟合
        gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        subsample=subsample, # 随机采样训练样本 训练实例的子采样比
        max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
        colsample_bytree=1, # 生成树时进行的列采样
        reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        #reg_alpha=0, # L1 正则项参数
        #scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
        #objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
        #num_class=10, # 类别数，多分类与 multisoftmax 并用
        n_estimators=n_estimators, #树的个数
        seed=1000 #随机种子
        #eval_metric= 'auc'
    )
    
#    X_train = X_train.append(X_test)
#    y_train = y_train.append(y_test)
#    print('X_train y_train shape is ', X_train.shape, y_train.shape)
    
#    xgbc.fit(X_train, y_train, eval_metric='auc')
    xgbc.fit(X_train, y_train, eval_metric=eval_metric_func)
    
    print("xgb accuracy on training set:", xgbc.score(X_train, y_train))
    outcome = xgbc.predict(X_test)
    score, score_a, score_b = get_AB_test_score(y_test.copy(), outcome)
    X_test_predicted = X_test.copy()
    X_test_predicted['predicted_is_risk'] = outcome
    X_test_predicted['real_is_risk'] = y_test
    
#    score = jd_score(y_test, outcome)
    print('in validation test get score ', score)
    
    
    time_str = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
    name_affix = ('xgb_' + time_str + '_' + str(round(score, 5)) + 
                  '_depth_' + str(max_depth) + 
                  '_learningrate_' + str(learning_rate) +
                  '_n_estimators_' + str(n_estimators) + 
                  '_subsample_' + str(subsample))
    store_feature_importances(xgbc, list(X_train.columns), name_affix)
    X_test_predicted.to_csv('./data/submission/tested_outcomes_' + name_affix + '.csv')
    with open('./model_dumps/xgb_' + name_affix + '.pkl', 'wb') as f:
        pickle.dump(xgbc, f)
    real_predicted_outcome = xgbc.predict(real_test_df)
    outcome_df = real_test_df.copy()
    outcome_df['is_risk'] = real_predicted_outcome
    outcome_df['is_risk'].to_csv('./data/submission/submission_'+ name_affix + '.csv')
    end_t = time.time()
    print('in training_with_xgboost, this round cost: ', end_t-start_t)
    
    

def training_with_gbdt(max_depth, learning_rate, n_estimators=600,
                       subsample=1.0, negative_weight_ratio=1.0):
    global X_train, y_train, X_test, y_test, real_test_df
    print('in training_with_gbdt, max_depth={}, learning_rate={} '
          'n_estimators={} subsample={} negative_weight_ratio={}'.format(
          max_depth, learning_rate, n_estimators, subsample, 
          negative_weight_ratio))
#    weight_arr = np.where(y_train==1, 1, 0.0283)
    positive_instances = y_train[y_train==1]
    negative_instances = y_train[y_train==0]
    negative_weight = (negative_weight_ratio*len(positive_instances)/
                       len(negative_instances))
    print('negative_weight is ', negative_weight)

    sample_weight = np.where(y_train==1, 1, negative_weight)
    gbdt = GradientBoostingClassifier(n_estimators=n_estimators, 
                                      max_depth=max_depth, 
                                      subsample=subsample,
                                      learning_rate=learning_rate,
                                      random_state=42,
                                      verbose=1)
    gbdt.fit(X_train, y_train, sample_weight=sample_weight)
#    gbdt.fit(X_train.iloc[:1000], y_train.iloc[:1000], sample_weight=sample_weight[:1000])
    
    print("accuracy on training set:", gbdt.score(X_train, y_train))
    outcome = gbdt.predict(X_test)
    score, score_a, score_b = get_AB_test_score(y_test, outcome)
#    score = jd_score(y_test, outcome)
    print('in validation test get score ', score)
    
    time_str = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
    name_affix = ('gbdt_' + time_str + '_' + str(round(score, 5)) + 
                  '_depth_' + str(max_depth) + 
                  '_learningrate_' + str(learning_rate) +
                  '_n_estimators_' + str(n_estimators) + 
                  '_subsample_' + str(subsample) +
                  '_negative_weight_ratio_' + str(negative_weight_ratio))
    store_feature_importances(gbdt, list(X_train.columns), name_affix)
    with open('./model_dumps/gbdt_' + name_affix + '.pkl', 'wb') as f:
        pickle.dump(gbdt, f)
    real_predicted_outcome = gbdt.predict(real_test_df)
    outcome_df = real_test_df.copy()
    outcome_df['is_risk'] = real_predicted_outcome
    outcome_df['is_risk'].to_csv('./data/submission/submission_'+ name_affix + '.csv')
    

def training_with_rf(n_estimators=4000, min_samples_leaf=3, sample_weight=None):
    global X_train, y_train, X_test, real_test_df
    print('in training_with_rf, n_estimators={}, min_samples_leaf={}'.format(
          n_estimators, min_samples_leaf))
    
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                n_jobs=1,
                                random_state=42,
                                min_samples_leaf=min_samples_leaf,
                                oob_score=True,
                                verbose=1
                                )
    rf.fit(X_train, y_train, sample_weight=sample_weight)
#    rf.fit(X_train.iloc[:1000], y_train.iloc[:1000], sample_weight=sample_weight[:1000])
    outcome = rf.predict(X_test)
    score = jd_score(y_test, outcome)
    print('in random forest validation test get score ', score)
    
    time_str = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
    name_affix = ('rf_' + time_str + '_' + str(round(score, 5)) + '_n_estimators_' + 
                  str(n_estimators) + '_min_samples_leaf_' + str(min_samples_leaf))
    store_feature_importances(rf, list(X_train.columns), name_affix)
    with open('./model_dumps/rf_' + name_affix + '.pkl', 'wb') as f:
        pickle.dump(rf, f)
    real_predicted_outcome = rf.predict(real_test_df)
    outcome_df = real_test_df.copy()
    outcome_df['is_risk'] = real_predicted_outcome
    outcome_df['is_risk'].to_csv('./data/submission/submission_'+ name_affix + '.csv')

def filter_out_features(dfs):
    filter_features = ['time', 'id', 'from_2015_1_1_minutes_num']
    if not isinstance(dfs, list):
        dfs = [dfs]
    for df in dfs:
        for feature in filter_features:
            df.drop(feature, axis=1, inplace=True)
    

if __name__=='__main__':
#    arr = np.array([12, 32, 33, 59, 67, 98, 102, 44, 55])
#    index_arr = np.array([1, 3, 4])
#    print(arr[index_arr])
#    print(time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime()))
#    print(jd_score111(0.45, 0.090))
#    sys.exit(0)  
      

    start_t = time.time()
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')    
    train_df = pd.read_csv('./data/train.csv', index_col='QuoteNumber',
                           parse_dates=['Original_Quote_Date'])
    
    y_train = train_df['QuoteConversion_Flag']
    X_train = train_df.drop('QuoteConversion_Flag', axis=1)

    
    X_test = pd.read_csv('./data/test.csv', index_col='QuoteNumber',
                           parse_dates=['Original_Quote_Date'])
    
    print(X_train.shape, y_train.shape, X_test.shape)
    sys.exit(0)
    
    X_train, X_test, y_train, y_test = train_test_split_new(
                         train_df.iloc[:, :-1], train_df.iloc[:, -1])
    
    real_test_df = pd.read_csv('./data/test_data.csv', index_col='rowkey',
                               parse_dates=['time'], date_parser=dateparse)
    
    filter_out_features([X_train, X_test, real_test_df])
    
#    X_train, y_train = inblance_preprocessing(X_train, y_train)
#    print('222 X_train y_train', X_train.shape, y_train.shape)
    y_test.to_csv('./data/submission/real_test_df.csv')
    
#    training_with_rf(sample_weight=weight_arr)

#    max_depth_list = [13]
#    learning_rate_list = [0.07, 0.09, 0.11, 0.13]
#    learning_rate_list = [0.09]

    max_depth_list = [5, 7, 9, 11, 13]
    learning_rate_list = [0.09, 0.11, 0.13, 0.15]
#    max_depth_list = [7]

    for depth in max_depth_list:
        for rate in learning_rate_list:
#            training_with_gbdt(depth, rate, n_estimators=2000,
#                               subsample=0.9, negative_weight_ratio=1.0)            
            training_with_xgboost(depth, rate, n_estimators=1200, 
                                  subsample=0.9)

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

#    pickle_in = open('./model_dumps/gbdt_gbdt_2017-11-17_11_16_42_0.4254_depth_13_learningrate_0.09_n_estimators_1000_subsample_0.9_negative_weight_ratio_1.0.pkl', 'rb')
#    gbdt = pickle.load(pickle_in)
#    
#    print("gbt accuracy on training set:", gbdt.score(X_train, y_train))
#    training_jd_score = jd_score(y_train, gbdt.predict(X_train))
#    print('training_jd_score is ', training_jd_score)
#    X_test_predicted_outcome = gbdt.predict(X_test)
#    score, score_a, score_b = get_AB_test_score(y_test, X_test_predicted_outcome)
    
#    score = jd_score(y_test, X_test_predicted_outcome)
#    print('score is ', score)
    
#    test_jd_score = jd_score(y_test, X_test_predicted_outcome)
#    print('test_jd_score is ', test_jd_score)
#    X_test['predicted_risk'] = X_test_predicted_outcome
#    X_test['is_risk'] = y_test
#    X_test[['predicted_risk', 'is_risk', 'id']].to_csv('./data/submission/test_predicted_outcomes.csv')   
    
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
    

#    pickle_in = open('./model_dumps/gbdt_2017-11-15_03_25_49.pkl', 'rb')
#    gbdt = pickle.load(pickle_in)
    
#    print("gbt accuracy on training set:", gbdt.score(X_train, y_train))
#    training_jd_score = jd_score(y_train, gbdt.predict(X_train))
#    print('training_jd_score is ', training_jd_score)
#    test_jd_score = jd_score(y_test, gbdt.predict(X_test))
#    print('test_jd_score is ', test_jd_score)    
    

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