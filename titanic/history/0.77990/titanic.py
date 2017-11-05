import sys
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from functools import reduce
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_moons
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from scipy.stats import skew
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

def convert_object(df):
    obj_df = df.select_dtypes(include=['object']).copy()
    for col_name in obj_df.columns:
        df[col_name] = df[col_name].fillna('NA')

def convert_int64(df):
    int64_df = df.select_dtypes(include=['int64']).copy()
    for col_name in int64_df.columns:
        df[col_name] = df[col_name].fillna(0)

def convert_float64(df):
    float64_df = df.select_dtypes(include=['float64']).copy()
    for col_name in float64_df.columns:
        df[col_name] = df[col_name].fillna(0.0)

def preprocess_Fare(df): # 将0 值换成 median值
    df['Fare'].replace(0, df['Fare'].median(), inplace=True)

def preprocess_Age(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    

def convert_dataframe(df):
    preprocess_Age(df)
    preprocess_Fare(df)
    convert_object(df)
    convert_int64(df)
    convert_float64(df)

def adjust_test_dataframe(test_df, train_df):
    for col_name in train_df.columns:
        if col_name not in test_df.columns:
            d_type = train_df[col_name].dtype
            if d_type==np.float64:
                test_df[col_name] = 0.
            elif d_type==np.int64:
                test_df[col_name] = 0
            elif d_type==np.object:
                test_df[col_name] = 'NA'

def preprocess_cabin(df):
    cabin_names_set = set()
    for val in df['Cabin']:
        cabin_names_set |= set(val.split())
    for name in cabin_names_set:
        col_name = 'Cabin' + '_' + name
        df[col_name] = df['Cabin'].apply(lambda s: 1 if name in s.split() else 0)
    del df['Cabin']
    return df
        
                

if __name__=='__main__':
    start_time = time.time()

    train_data_frame = pd.read_csv('./train.csv')
    test_data_frame = pd.read_csv('./test.csv')
    
#    print('train_data_frame', list(train_data_frame.columns))
#    print('train_data_frame', train_data_frame['Cabin_C123'])
    
#    used_cols = list(set(train_data_frame.columns) -
#                     set(['PassengerId', 'Survived', 'Name', 
#                          'Ticket', 'Cabin']))
    used_cols = list(set(train_data_frame.columns) -
                     set(['PassengerId', 'Survived', 'Name', 'Ticket']))
    features_data_frame = train_data_frame[used_cols]
    target_data_frame = train_data_frame['Survived']
    test_data_frame = test_data_frame[used_cols]    
    
#    print('year that was build', train_data_frame['GarageYrBlt'].dtype)
#    print('year that was build', test_data_frame['GarageYrBlt'])
#    train_data_frame['SalePrice'].value_counts().plot(kind='barh')
#    print(train_data_frame.dtypes)
    
    convert_dataframe(features_data_frame)
    convert_dataframe(test_data_frame)
    
    print('features_data_frame is ', list(features_data_frame['Age']))
    
    concated_dataframe = pd.concat([features_data_frame, test_data_frame])
    preprocess_cabin(concated_dataframe)
    concated_dataframe = pd.get_dummies(concated_dataframe)
#    concated_dataframe = concated_dataframe.select_dtypes(include=['float', 'int']).copy()
    features_data_frame = concated_dataframe[:len(features_data_frame)]
#    print('features_dataframe.dtypes is ', features_dataframe.dtypes)
#    print('features_dataframe.columns is ', list(features_data_frame.columns))
    # 'Embarked_C', 'Embarked_NA', 'Embarked_Q', 'Embarked_S'
    test_data_frame = concated_dataframe[len(features_data_frame):]
#    print('test_dataframe.shape is ', test_data_frame.shape)


#    features_data_frame = pd.get_dummies(features_data_frame)
#    test_data_frame = pd.get_dummies(test_data_frame)
#    adjust_test_dataframe(features_data_frame, train_data_frame)
    
#    features_cols = list(set(train_data_frame.columns)-set(['Survived']))

#    print(data_dummies.dtypes)
#    col_names = list(data_dummies.columns)
#    train_cols = list(set(train_data_frame.columns)-
#                      set(['PassengerId', 'Survived', 'Name', 'Ticket']))
    
#    X_train, X_test, y_train, y_test = train_test_split(
#                                            train_data_frame[train_cols], 
#                                            train_data_frame['SalePrice'], 
#                                            random_state=42)

######################################################################
            
#    lr = LinearRegression().fit(X_train, y_train)
                                            
#    best_ratio = 0
#    best_score = -1000
#    scores_mean_list = []
#    ratio_list = []
#    for ratio in range(10, 100, 10):
#        print('ratio is ', ratio)
#        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
#        rf = RandomForestRegressor(n_estimators=1000,
#                               max_features=int(len(train_cols)*ratio/100),
#                               max_depth=4,                               
#                               n_jobs=4)
#        scores = cross_val_score(rf, train_data_frame[train_cols], 
#                                 train_data_frame['SalePrice'], 
#                                 cv=kfold)
#        print('ratio is ', ratio, 'scores mean is ', scores.mean())
#        scores_mean_list.append(scores.mean())
#        ratio_list.append(ratio)
#        if best_score < scores.mean():
#            best_score = scores.mean()
#            best_ratio = ratio

#    plt.plot(ratio_list, scores_mean_list)
#    plt.show()
    print('training start...')
#    rf = RandomForestRegressor(n_estimators=10000, 
#                               max_features=int(len(train_cols)*best_ratio/100),
#                               max_depth=4,
#                               n_jobs=4).fit(
#                               train_data_frame[train_cols],
#                               train_data_frame['SalePrice'])

###########################################################################

    param_grid = {'n_estimators': [3000],
                  'learning_rate': [0.001, 0.001, 0.1],
                  'max_depth': [2, 3, 4, 5]}
                  
    grid_search = GridSearchCV(GradientBoostingClassifier(random_state=5), 
                               param_grid, cv=5)
    grid_search.fit(features_data_frame, target_data_frame)
    test_score = grid_search.score(features_data_frame, target_data_frame)
    outcome = list(grid_search.predict(test_data_frame))
    
    print('svr grid_search.best_params_ is ', grid_search.best_params_)
    print('svr grid_search.best_score_ is ', grid_search.best_score_)
    print('best score is ', test_score)
    

#    gbr = GradientBoostingClassifier(n_estimators=1000, max_depth=4, 
#                                     learning_rate=0.07,
#                                     random_state=0)
#    scores = cross_val_score(gbr, features_data_frame, target_data_frame, cv=5)
#    print('scores mean is ', scores.mean())
#                               
##    print('features_data_frame is ', features_data_frame['Sex'])
#    gbr.fit(features_data_frame, target_data_frame)
#    print("accuracy on training set:", gbr.score(features_data_frame, 
#                                                 target_data_frame))
#
#    print('length of test_data_frame ', len(test_data_frame))
#    outcome = list(gbr.predict(test_data_frame))
#    print('length of outcome is ', len(outcome))
    
#######################################################################
    
#    lr = LogisticRegression()
#    lr.fit(features_data_frame, target_data_frame)
#    print("accuracy on training set:", lr.score(features_data_frame, 
#                                                 target_data_frame))
#
#    print('length of test_data_frame ', len(test_data_frame))
#    outcome = list(lr.predict(test_data_frame))
#    print('length of outcome is ', len(outcome))
    
#######################################################################
    
#    param_grid = {'kernel': ["rbf"],
#                  'C'     : np.logspace(-5, 5, num=11, base=10.0),
#                  'gamma' : np.logspace(-5, 5, num=11, base=10.0)}
#    df_concated = pd.concat([features_data_frame, 
#                             test_data_frame])
#    scaler = StandardScaler().fit(df_concated)
#    x_train_scaled = scaler.transform(features_data_frame)
#    x_test_scaled = scaler.transform(test_data_frame)
#    
#    grid_search = GridSearchCV(SVC(), 
#                               param_grid, cv=5)
#    grid_search.fit(features_data_frame, target_data_frame)
#    test_score = grid_search.score(features_data_frame, target_data_frame)
#    outcome = list(grid_search.predict(test_data_frame))
#    
#    print('svr grid_search.best_params_ is ', grid_search.best_params_)
#    print('svr grid_search.best_score_ is ', grid_search.best_score_)
#    print('best score is ', test_score)



#######################################################################

    outcome_data_frame = pd.DataFrame(
                         {'PassengerId': list(range(892, 1310)), 
                          'Survived': outcome
                         }, index=None)
                         
    outcome_data_frame = outcome_data_frame.set_index('PassengerId')
    outcome_data_frame.to_csv('./tsg_outcome.csv')

    end_time = time.time()    
    print('time cost is ', end_time-start_time)
