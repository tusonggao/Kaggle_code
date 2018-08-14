# https://www.kaggle.com/apapiu/regularized-linear-models

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###########################
import xgboost as xgb
###########################
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV,  LassoLarsCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import matplotlib
from sklearn import linear_model
import numpy as np
import pandas as pd
from functools import reduce

from sklearn.datasets import make_moons
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
#import mglearn
import xgboost as xgb
import time
import sklearn
from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.metrics import mean_squared_error


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
#prices = pd.DataFrame({"price":train["SalePrice"], 
#                       "log(price + 1)":np.log1p(train["SalePrice"]),
#                       'TotalBsmtSF': train['TotalBsmtSF'],
#                       "log(TotalBsmtSF)":np.log1p(train["TotalBsmtSF"])})
#train.hist(column=['TotalBsmtSF', 'SalePrice'], bins=20, sharex=True)

train["SalePrice"] = np.log1p(train["SalePrice"])
print('all_data.dtypes is ', all_data.dtypes)
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
#skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
print('skewed_feats is ', skewed_feats)

#train["SalePrice"] = np.log1p(train["SalePrice"])
#numeric_features = df.select_dtypes(include=[np.number]).columns.values
#
#print('all_data.dtypes is ', all_data.dtypes)
#numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
#skewed_feats = train[numeric_features].apply(lambda x: skew(x.dropna())) #compute skewness
#skewed_feats = skewed_feats[skewed_feats > 0.75]
#skewed_feats = skewed_feats.columns
#print('skewed_feats is ', skewed_feats)

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())
#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, 
                                   scoring="neg_mean_squared_error", cv = 5))
    return rmse

model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
print('cv_ridge.min() is ', cv_ridge.min())

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
print('rmse_cv(model_lasso).mean() is ', rmse_cv(model_lasso).mean())
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + 
      " variables and eliminated the other " +  
       str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")

dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)
xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))
predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")

preds = 0.7*lasso_preds + 0.3*xgb_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("./data/ridge_sol.csv", index = False)







































