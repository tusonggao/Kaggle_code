import pandas as pd
import math
import numpy as np
import os
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def rmse_new(y_true, y_pred):
    if np.sum(y_pred<0)>0:
        print('negative exits.')
    else:
        print('negative not exits.')
    y_pred = np.where(y_pred>0, y_pred, 0)
    return 'RMSE', np.sqrt(mean_squared_error(y_true, y_pred)), False

def rmse(y_true, y_pred):
    return 'RMSE', np.sqrt(mean_squared_error(y_true, y_pred)), False


start_t = time.time()

train_file='./data/tap_fun_train.csv'
test_file='./data/tap_fun_test.csv'
train_data=pd.read_csv(train_file)
test_data=pd.read_csv(test_file)

train_data=train_data.fillna(0)
test_data=test_data.fillna(0)

print(train_data.shape, test_data.shape)
drop = ['user_id', 'register_time']

train_idx=train_data[drop]
test_idx=test_data[drop]
train_data=train_data.drop(drop, axis=1)
test_data=test_data.drop(drop, axis=1)

cols = ['pay_price']
train_X = train_data[cols]
train_y = train_data.pop('prediction_pay_price')

lr = LinearRegression()

X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_X, 
            train_y, test_size=0.25, random_state=42)
print('X_train_new.shape is {}, X_val_new.shape is {}'.format(
      X_train_new.shape, X_val_new.shape))

lr.fit(X_train_new, y_train_new)
y_pred = lr.predict(X_val_new)
rmse_val = rmse(y_pred, y_val_new)[1]
rmse_val_new = rmse_new(y_pred, y_val_new)[1]
print('rmse_val is ', rmse_val, 'rmse_val_new is ', rmse_val_new)

lr.fit(train_X, train_y)
y_prob = lr.predict(test_data[cols])
test_idx['prediction_pay_price'] = np.round(y_prob, 5)
print(test_idx.prediction_pay_price.value_counts())
test_idx[['user_id','prediction_pay_price']].to_csv("./outcome/submission_example_new.csv",
        index=False)

print('end of prog, cost time: ', time.time()-start_t)


