import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

df = pd.read_csv('C:/D_Disk/data_competition/model_stacking/input.csv')
df.index.name = 'Id'

df.loc[df['cand_pty_affiliation']=='DEM', 'cand_pty_affiliation'] = 1
df.loc[df['cand_pty_affiliation']=='REP', 'cand_pty_affiliation'] = 0

test_size = 0.7
msk_1 = np.random.rand(len(df)) < test_size
test_df = df[msk_1]
train_df = df[~msk_1]
print('train_df.shape test_df.shape is ', train_df.shape, test_df.shape)

private_size = 0.6
msk_2 = np.random.rand(len(test_df)) < private_size
private_test_df = test_df[msk_2]
public_train_df = test_df[~msk_2]

private_test_df.to_csv('C:/D_Disk/data_competition/model_stacking/input_test_private.csv')
public_train_df.to_csv('C:/D_Disk/data_competition/model_stacking/input_test_public.csv')

print('test_df.columns is ', test_df.columns)
test_df.drop(['cand_pty_affiliation'], axis=1, inplace=True)

train_df.to_csv('C:/D_Disk/data_competition/model_stacking/input_train.csv')
test_df.to_csv('C:/D_Disk/data_competition/model_stacking/input_test.csv')
