import pandas as pd
import numpy as np


if __name__=='__main__':
    print('Hello world!')
    train_df = pd.read_csv('./dataset/train.csv')
    test_df = pd.read_csv('./dataset/test.csv')
    print(train_df.head())
    print(test_df.head())
    
    
    
