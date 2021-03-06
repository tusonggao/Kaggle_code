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
    df['Fare'].replace(0.0, np.nan, inplace=True)
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

def preprocess_Age(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())


def preprocess_Cabin_1(df): #处理方法一
    df['Cabin'].fillna('', inplace=True)
    cabin_names_set = set()
    for val in df['Cabin']:
        cabin_names_set |= set(val.split())
    for name in cabin_names_set:
        col_name = 'Cabin' + '_' + name
        df[col_name] = df['Cabin'].apply(lambda s: 1 if name in s.split() else 0)
    del df['Cabin']
    return df

def get_cabin_alphabet_set(s):
    cabin_alphabet_set = set()
    for c in s:
        if c.isalpha():
            cabin_alphabet_set.add(c)
    return cabin_alphabet_set

def preprocess_Cabin_2(df): #处理方法二
    df['Cabin'].fillna('', inplace=True)
    cabin_names_set = set()
    cabin_alpha_set = set()
    for val in df['Cabin']:
        cabin_names_set |= set(val.split())
        cabin_alpha_set |= get_cabin_alphabet_set(val)
    for name in cabin_names_set:
        col_name = 'Cabin' + '_' + name
        df[col_name] = df['Cabin'].apply(lambda s: 1 if name in s.split() else 0)
    for alpha in cabin_alpha_set:
        col_name = 'Cabin' + '_' + alpha
        df[col_name] = df['Cabin'].apply(lambda s: 1 if alpha in s else 0)
    del df['Cabin']
    return df

def preprocess_Pclass(df):
    df['Pclass'] = df['Pclass'].apply(lambda pn: str(pn))

def convert_dataframe(df):
    preprocess_Cabin_2(df)
    preprocess_Age(df)
    preprocess_Fare(df)
    preprocess_Pclass(df)
#    convert_object(df)
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
                
if __name__=='__main__':
    start_time = time.time()

    train_data_frame = pd.read_csv('./train.csv')
#    count_nan = train_data_frame['Ticket'].notnull().sum()
#    print('count_nan is ', count_nan)
#    sys.exit(1)
    test_data_frame = pd.read_csv('./test.csv')
    
    print(train_data_frame[['Age', 'Survived']].corr())
    
#    used_cols = list(set(train_data_frame.columns) -
#                     set(['PassengerId', 'Survived', 'Name', 'Ticket']))
#    cols_to_drop = ['PassengerId', 'Survived', 'Name', 'Ticket']
    cols_to_drop = ['PassengerId', 'Survived', 'Name']
    features_data_frame = train_data_frame.drop(cols_to_drop, axis=1, errors='ignore')
    target_data_frame = train_data_frame['Survived']
    test_data_frame = test_data_frame.drop(cols_to_drop, axis=1, errors='ignore')
    
    
    
    
#    concated_dataframe = pd.concat([features_data_frame, test_data_frame])
    concated_dataframe = features_data_frame.append(test_data_frame)
    convert_dataframe(concated_dataframe)
    concated_dataframe = pd.get_dummies(concated_dataframe)
    
#    concated_dataframe = concated_dataframe.select_dtypes(include=['float', 'int']).copy()
    features_data_frame = concated_dataframe[:len(features_data_frame)]
    test_data_frame = concated_dataframe[len(features_data_frame):]
#    print('features_data_frame is ', list(features_data_frame.columns))
#    print('features_data_frame is ', list(features_data_frame['Fare'])[295:307])
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

#svr grid_search.best_params_ is  {'n_estimators': 5000, 'learning_rate': 0.1, 'max_depth': 6}
#svr grid_search.best_score_ is  0.846240179574
#best score is  0.997755331089
#time cost is  7278.848999977112

    param_grid = {'n_estimators': [5000],
                  'learning_rate': [0.001, 0.001, 0.1],
                  'max_depth': [2, 4, 6, 8, 10, None]}
                  
    grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), 
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

######################################################################

#    param_grid = {'n_estimators': [5000, 8000],
#                  'max_depth': [8, 15, 20, None]}
#                  
#    grid_search = GridSearchCV(RandomForestClassifier(random_state=5), 
#                               param_grid, cv=5)
#    grid_search.fit(features_data_frame, target_data_frame)
#    test_score = grid_search.score(features_data_frame, target_data_frame)
#    outcome = list(grid_search.predict(test_data_frame))
#    
#    print('svr grid_search.best_params_ is ', grid_search.best_params_)
#    print('svr grid_search.best_score_ is ', grid_search.best_score_)
#    print('best score is ', test_score)
    
#######################################################################
    
    ### Logistic Regression 需要feature rescaling
    
#    df_concated = pd.concat([features_data_frame, test_data_frame])
#    scaler = StandardScaler().fit(df_concated)
#    features_data_frame = scaler.transform(features_data_frame)
#    test_data_frame = scaler.transform(test_data_frame)
#    
#    param_grid = {'penalty': ['l1', 'l2'],
#                  'C': [0.01, 0.1, 1, 10, 100],
#                  'solver': ['liblinear']}
#                  
#    grid_search = GridSearchCV(LogisticRegression(random_state=5), 
#                               param_grid, cv=5)
#    grid_search.fit(features_data_frame, target_data_frame)
#    test_score = grid_search.score(features_data_frame, target_data_frame)
#    outcome = list(grid_search.predict(test_data_frame))
#    
#    print('svr grid_search.best_params_ is ', grid_search.best_params_)
#    print('svr grid_search.best_score_ is ', grid_search.best_score_)
#    print('best score is ', test_score)
    
    
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
#    df_concated = pd.concat([features_data_frame, test_data_frame])
#    scaler = StandardScaler().fit(df_concated)
#    x_train_scaled = scaler.transform(features_data_frame)
#    x_test_scaled = scaler.transform(test_data_frame)
#    
#    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
#    grid_search.fit(x_train_scaled, target_data_frame)
#    test_score = grid_search.score(x_train_scaled, target_data_frame)
#    outcome = list(grid_search.predict(x_test_scaled))
#    
#    print('svr grid_search.best_params_ is ', grid_search.best_params_)
#    print('svr grid_search.best_score_ is ', grid_search.best_score_)
#    print('best score is ', test_score)


#######################################################################

    outcome_data_frame = pd.DataFrame(
                         {'PassengerId': list(range(892, 1310)), 
                          'Survived': outcome
                         }, index=None)
                         
#    outcome_data_frame = pd.DataFrame(
#                         {'PassengerId': list(range(892, 1310)), 
#                          'Survived': list(np.random.randint(0, 2, size=418))
#                         }, index=None)
                         
    outcome_data_frame = outcome_data_frame.set_index('PassengerId')
    outcome_data_frame.to_csv('./tsg_outcome.csv')

    end_time = time.time()    
    print('time cost is ', end_time - start_time)
    
    
#######################################################################
