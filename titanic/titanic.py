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
from sklearn.tree import DecisionTreeClassifier
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

def preprocess_Name(df):
    df['First_Name'] = df['Name'].str.extract('^(.+?),').str.strip()
    df['Last_Name'] = df['Name'].str.split("\.").str[1].str.strip()
    df['Last_Name'] = df['Last_Name'].str.strip("\([^)]*\)")
    df['Last_Name'].fillna(df['Name'].str.split("\.").str[1].str.strip())
    del df['Name']

def convert_dataframe(df):
    preprocess_Cabin_2(df)
    preprocess_Age(df)
    preprocess_Fare(df)
#    preprocess_Pclass(df) 
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
    print('Hello world')

    train_data_frame = pd.read_csv('./train.csv')
    test_data_frame = pd.read_csv('./test.csv')
    train_data_frame.set_index(['PassengerId'], inplace=True)
    test_data_frame.set_index(['PassengerId'], inplace=True)
    
#    guess_outcome = test_data_frame['Sex']=='female'
#    print('test_data_frame sex value_counts \n', test_data_frame['Sex'].value_counts())    
#    outcome_data_frame = pd.DataFrame(
#                         {'Survived': [1 if x==True else 0 for x in guess_outcome]}, 
#                         index=test_data_frame.index.values)
#    outcome_data_frame.to_csv('./tsg_outcome.csv', index_label='PassengerId')
#    end_time = time.time()    
#    print('time cost is ', end_time - start_time)    
#    sys.exit(0)
    
#    cols_to_drop = ['Survived']
    X_train = train_data_frame.drop(['Survived'], axis=1)
    y_train = train_data_frame['Survived']
    X_test = test_data_frame
    
    
#    concated_dataframe = pd.concat([features_data_frame, test_data_frame])
    concated_dataframe = X_train.append(X_test)
    convert_dataframe(concated_dataframe)
    concated_dataframe = pd.get_dummies(concated_dataframe)
    
#    concated_dataframe = concated_dataframe.select_dtypes(include=['float', 'int']).copy()
    X_train = concated_dataframe[:len(X_train)]
    print('X_train.shape is ', X_train.shape)
    
    X_test = concated_dataframe[len(X_train):]
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

#svr grid_search.best_params_ is  {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 5000}
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
    

#    gbr = GradientBoostingClassifier(n_estimators=300, max_depth=2, 
#                                     learning_rate=0.01,
#                                     random_state=42)
#    scores = cross_val_score(gbr, X_train, y_train, cv=5)
#    print('scores mean is ', scores.mean())
                               
#    gbr.fit(X_train, y_train)
#    print("accuracy on training set:", gbr.score(X_train, y_train))
#    outcome = gbr.predict(X_test)
#    gbr_prob = gbr.predict_proba(X_test)
#    print('gbr_prob is ', gbr_prob)
#    print('shape of gbr_prob is ', gbr_prob.shape)
#    gbr_prob_frame = pd.DataFrame({'0': gbr_prob[:, 0], 
#                                   '1':gbr_prob[:, 1],
#                                   'outcome': outcome},
#                                   index=X_test.index.values)
#    gbr_prob_frame.to_csv('./gbr_prob_frame.csv', index_label='PassengerId')

######################################################################

    param_grid = {'n_estimators': [1000],
                  'max_depth': [8, 15, 30, None],
                  'max_features': [0.5, 'auto', None],
                  'min_samples_leaf': [3, 4, 5, 6],
                  'oob_score': [True]}
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                               param_grid, cv=5, n_jobs=2)
    grid_search.fit(X_train, y_train)
    test_score = grid_search.score(X_train, y_train)
    outcome = grid_search.predict(X_test)
    
    
    print('svr grid_search.best_params_ is ', grid_search.best_params_)
    print('svr grid_search.best_score_ is ', grid_search.best_score_)
    print('best score is ', test_score)
    
#    rf = RandomForestClassifier(n_estimators=1000,
#                                n_jobs=2,
#                                random_state=42,
#                                min_samples_leaf=5,
#                                oob_score=True
#                                )
#    scores = cross_val_score(rf, X_train, y_train, cv=5)
#    print('scores mean is ', scores.mean())
                               
#    rf.fit(X_train, y_train)
#    print("accuracy on training set:", rf.oob_score_)
#
#    outcome = rf.predict(X_test)
#    rf_prob = rf.predict_proba(X_test)
#    print('shape of rf_prob is ', rf_prob.shape)
#    rf_prob_frame = pd.DataFrame({'0': rf_prob[:, 0], 
#                                  '1': rf_prob[:, 1],
#                                  'outcome': outcome},
#                                  index=X_test.index.values)
#    rf_prob_frame.to_csv('./rf_prob_frame.csv', index_label='PassengerId')
    
    
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

#    sum_prob = gbr_prob + rf_prob
#    outcome = sum_prob[:, 0] < sum_prob[:, 1]
#    sum_prob_frame = pd.DataFrame({'0': sum_prob[:, 0], '1':sum_prob[:, 1],
#                                   'outcome': outcome},
#                                  index=X_test.index.values)
#    sum_prob_frame.to_csv('./sum_prob_frame.csv', index_label='PassengerId')

    outcome_data_frame = pd.DataFrame(data = outcome, 
                                      index = X_test.index.values, 
                                      columns = ['Survived'])
    
#    outcome_data_frame = pd.DataFrame(data = sum_prob['outcome'], 
#                                      index = X_test.index.values, 
#                                      columns = ['Survived'])   
                         
#    outcome_data_frame = pd.DataFrame(
#                         {'Survived': [1 if x==True else 0 for x in outcome]}, 
#                         index=X_test.index.values)
                         
#    outcome_data_frame = outcome_data_frame.set_index('PassengerId')
    outcome_data_frame.to_csv('./tsg_outcome.csv', index_label='PassengerId')

    end_time = time.time()    
    print('time cost is ', end_time - start_time)
    
    
#######################################################################
