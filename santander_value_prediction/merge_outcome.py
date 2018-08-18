import numpy as np
import pandas as pd


df_lgbm = pd.read_csv('./outcome/sub_.csv', index_col=0)

df_leak = pd.read_csv('./outcome/outcome_.csv', index_col=0)

df_lgbm.loc[df_leak.index, 'target'] = df_leak['target']

df_lgbm.to_csv('./outcome/merged_outcome.csv', head=['target'])