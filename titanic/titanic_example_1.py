# https://www.kaggle.com/c/titanic/leaderboard
# https://www.kaggle.com/rafalplis/my-approach-to-titanic-competition

import sys
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import matplotlib
import seaborn as sns

import sklearn

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingClassifier

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_breast_cancer

from sklearn import ensemble
from sklearn import model_selection

from sklearn.tree import DecisionTreeClassifier

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from functools import reduce

from sklearn.datasets import make_moons

from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from scipy.stats import skew

dfTrain = pd.read_csv("./train.csv") # importing train dataset
dfTest = pd.read_csv("./test.csv") # importing test dataset

print(dfTrain.info())

dfTrain.set_index(['PassengerId'],inplace=True)
dfTest.set_index(['PassengerId'],inplace=True)

#dfTrain.groupby('Pclass').Survived.mean().plot(kind='bar')
#dfTrain.groupby('Sex').Survived.mean().plot(kind='bar')
print('dfTrain.shape is ', dfTrain.shape)
print(dfTrain.count())

#sns.factorplot("Sex", "Survived", hue="Pclass", data=dfTrain)

dfFull = pd.concat([dfTrain,dfTest])
dfFull['Sex'] = dfFull['Sex'].map({'male': 0, 'female': 1}).astype(int)
dfTrain = dfFull.loc[1:891,:]
dfTest = dfFull.loc[892:,:]

dtree = DecisionTreeClassifier()
X_train = dfTrain[['Pclass','Sex']]
y = dfTrain['Survived']
X_test = dfTest[['Pclass','Sex']]
dtree.fit(X_train, y)
prediction = dtree.predict(X_test)
dfPrediction = pd.DataFrame(data = prediction,
                            index = dfTest.index.values,
                            columns = ['Survived'])
contentTestPredObject1 = dfPrediction.to_csv('./tsg_outcome.csv', 
                                             index_label='PassengerId')



























