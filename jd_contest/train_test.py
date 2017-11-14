import numpy as np
np.random.seed(2017)

import sys
import time
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle as skl_shuffle


def jd_score(y_true, y_predicted, beta=0.1):
    precision = metrics.precision_score(y_true, y_predicted)
    recall = metrics.recall_score(y_true, y_predicted)
    accuracy = metrics.accuracy_score(y_true, y_predicted)
    print('precision, recall, accuracy is ', precision, recall, accuracy)
    score = (1 + beta**2)*(precision*recall)/(beta**2*precision + recall)
    return score
    
def jd_score111(precision, recall, beta=0.1):
#    print('precision, recall, accuracy is ', precision, recall, accuracy)
    score = (1 + beta**2)*(precision*recall)/(beta**2*precision + recall)
    return score

def inblance_preprocessing(data_df, label_df):
    data_df = pd.concat([data_df, label_df], axis=1)
    positive_instances = data_df[data_df['is_risk']==1]
    negative_instances = data_df[data_df['is_risk']==0]
    print('positive_instances negative_instances len is ', 
          len(positive_instances), len(negative_instances),
          'positive_instances negative_instances shape is ', 
          positive_instances.shape, negative_instances.shape)

    all_instances =  negative_instances
    for _ in range(10):
        all_instances = all_instances.append(positive_instances)
    print('all_instances len is ', len(all_instances),
          'shape is ', all_instances.shape)
    all_instances = skl_shuffle(all_instances)

    return all_instances.iloc[:, :-1], all_instances.iloc[:, -1]
    

if __name__=='__main__':
    print(jd_score111(0.91, 0.93))
    sys.exit(0)    


    start_t = time.time()
    train_df = pd.read_csv('./data/train_data.csv', index_col='rowkey')
    X_train, X_test, y_train, y_test = train_test_split(
                                           train_df.iloc[:, :-1], 
                                           train_df.iloc[:, -1],
                                           test_size=0.25,
                                           random_state=42)
    print('111 X_train X_test', X_train.shape, X_test.shape, 
          y_train.shape, y_test.shape)
    
    X_train, y_train = inblance_preprocessing(X_train, y_train)
    print('222 X_train y_train', X_train.shape, y_train.shape)
    
    
    
#    gbr = GradientBoostingClassifier(n_estimators=1000, max_depth=13, 
#                                     learning_rate=0.05,
#                                     random_state=42,
#                                     verbose=1)
    
#    gbr = GradientBoostingClassifier(n_estimators=100, max_depth=5, 
#                                     learning_rate=0.05,
#                                     random_state=42,
#                                     verbose=1)        
#    gbr.fit(X_train.iloc[:4000], y_train.iloc[:4000])

#    gbr.fit(X_train, y_train)
#    print("accuracy on training set:", gbr.score(X_train, y_train))
#    
#    with open('./model_dumps/gbdt.pkl', 'wb') as f:
#        pickle.dump(gbr, f)
#    outcome = gbr.predict(X_test)
#    score = jd_score(y_test, outcome)
#    print('in validation test get score ', score)
#    
    real_test_df = pd.read_csv('./data/test_data.csv', index_col='rowkey')
#    real_predicted_outcome = gbr.predict(real_test_df)
#    real_test_df['is_risk'] = real_predicted_outcome
#    real_test_df.to_csv('./data/outcomes.csv')

    train_id_set = set(X_train['id'])
    test_id_set = set(X_test['id'])
    real_test_id_set = set(real_test_df['id'])
    print('train_id_set > test_id_set: ', train_id_set > test_id_set)
    print('train_id_set > real_test_id_set: ', train_id_set > real_test_id_set)
   

    
    pickle_in = open('./model_dumps/gbdt.pkl', 'rb')
    gbt = pickle.load(pickle_in)
    print("gbt accuracy on training set:", gbt.score(X_train, y_train))
    training_jd_score = jd_score(y_train, gbt.predict(X_train))
    print('training_jd_score is ', training_jd_score)
    test_jd_score = jd_score(y_test, gbt.predict(X_test))
    print('test_jd_score is ', test_jd_score)

    end_t = time.time()
    print('total cost time is ', end_t-start_t)
    
    recall = 1
    precision = 0.5
    print(jd_score111(0.5, 1))
    
    
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