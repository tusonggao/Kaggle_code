import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
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
from scipy.stats import skew
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
#import mglearn
import xgboost as xgb
import time
import sklearn

from sklearn.metrics import mean_squared_error

def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))

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

def convert_dataframe(df):
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
                
def anova(frame, features):
    anv = pd.DataFrame()
    anv['features'] = features
    pvals = []
    for c in features:
           samples = []
           for cls in frame[c].unique():
                  s = frame[frame[c] == cls]['SalePrice'].values
                  samples.append(s)
           pval = stats.f_oneway(*samples)[1]
           pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort(columns=['pval'])

def boxplot(x, y, **kwargs):
    sns.boxplot(x=x,y=y)
    x = plt.xticks(rotation=90)
    

le = LabelEncoder()
def factorize(data, var, fill_na = None):
    if fill_na is not None:
        data[var].fillna(fill_na, inplace=True)
    le.fit(data[var])
    data[var] = le.transform(data[var])
    return data
    

def onehot(onehot_df, df, column_name, fill_na):
       onehot_df[column_name] = df[column_name]
       if fill_na is not None:
            onehot_df[column_name].fillna(fill_na, inplace=True)

       dummies = pd.get_dummies(onehot_df[column_name], prefix="_"+column_name)
       onehot_df = onehot_df.join(dummies)
       onehot_df = onehot_df.drop([column_name], axis=1)
       return onehot_df

def munge_onehot(df):
       onehot_df = pd.DataFrame(index = df.index)

       onehot_df = onehot(onehot_df, df, "MSSubClass", None)
       onehot_df = onehot(onehot_df, df, "MSZoning", "RL")
       onehot_df = onehot(onehot_df, df, "LotConfig", None)
       onehot_df = onehot(onehot_df, df, "Neighborhood", None)
       onehot_df = onehot(onehot_df, df, "Condition1", None)
       onehot_df = onehot(onehot_df, df, "BldgType", None)
       onehot_df = onehot(onehot_df, df, "HouseStyle", None)
       onehot_df = onehot(onehot_df, df, "RoofStyle", None)
       onehot_df = onehot(onehot_df, df, "Exterior1st", "VinylSd")
       onehot_df = onehot(onehot_df, df, "Exterior2nd", "VinylSd")
       onehot_df = onehot(onehot_df, df, "Foundation", None)
       onehot_df = onehot(onehot_df, df, "SaleType", "WD")
       onehot_df = onehot(onehot_df, df, "SaleCondition", "Normal")

       #Fill in missing MasVnrType for rows that do have a MasVnrArea.
       temp_df = df[["MasVnrType", "MasVnrArea"]].copy()
       idx = (df["MasVnrArea"] != 0) & ((df["MasVnrType"] == "None") | (df["MasVnrType"].isnull()))
       temp_df.loc[idx, "MasVnrType"] = "BrkFace"
       onehot_df = onehot(onehot_df, temp_df, "MasVnrType", "None")

       onehot_df = onehot(onehot_df, df, "LotShape", None)
       onehot_df = onehot(onehot_df, df, "LandContour", None)
       onehot_df = onehot(onehot_df, df, "LandSlope", None)
       onehot_df = onehot(onehot_df, df, "Electrical", "SBrkr")
       onehot_df = onehot(onehot_df, df, "GarageType", "None")
       onehot_df = onehot(onehot_df, df, "PavedDrive", None)
       onehot_df = onehot(onehot_df, df, "MiscFeature", "None")
       onehot_df = onehot(onehot_df, df, "Street", None)
       onehot_df = onehot(onehot_df, df, "Alley", "None")
       onehot_df = onehot(onehot_df, df, "Condition2", None)
       onehot_df = onehot(onehot_df, df, "RoofMatl", None)
       onehot_df = onehot(onehot_df, df, "Heating", None)

       # we'll have these as numerical variables too
       onehot_df = onehot(onehot_df, df, "ExterQual", "None")
       onehot_df = onehot(onehot_df, df, "ExterCond", "None")
       onehot_df = onehot(onehot_df, df, "BsmtQual", "None")
       onehot_df = onehot(onehot_df, df, "BsmtCond", "None")
       onehot_df = onehot(onehot_df, df, "HeatingQC", "None")
       onehot_df = onehot(onehot_df, df, "KitchenQual", "TA")
       onehot_df = onehot(onehot_df, df, "FireplaceQu", "None")
       onehot_df = onehot(onehot_df, df, "GarageQual", "None")
       onehot_df = onehot(onehot_df, df, "GarageCond", "None")
       onehot_df = onehot(onehot_df, df, "PoolQC", "None")
       onehot_df = onehot(onehot_df, df, "BsmtExposure", "None")
       onehot_df = onehot(onehot_df, df, "BsmtFinType1", "None")
       onehot_df = onehot(onehot_df, df, "BsmtFinType2", "None")
       onehot_df = onehot(onehot_df, df, "Functional", "Typ")
       onehot_df = onehot(onehot_df, df, "GarageFinish", "None")
       onehot_df = onehot(onehot_df, df, "Fence", "None")
       onehot_df = onehot(onehot_df, df, "MoSold", None)

       # Divide  the years between 1871 and 2010 into slices of 20 years
       year_map = pd.concat(pd.Series("YearBin" + str(i+1), index=range(1871+i*20,1891+i*20))  for i in range(0, 7))
       yearbin_df = pd.DataFrame(index = df.index)
       yearbin_df["GarageYrBltBin"] = df.GarageYrBlt.map(year_map)
       yearbin_df["GarageYrBltBin"].fillna("NoGarage", inplace=True)
       yearbin_df["YearBuiltBin"] = df.YearBuilt.map(year_map)
       yearbin_df["YearRemodAddBin"] = df.YearRemodAdd.map(year_map)

       onehot_df = onehot(onehot_df, yearbin_df, "GarageYrBltBin", None)
       onehot_df = onehot(onehot_df, yearbin_df, "YearBuiltBin", None)
       onehot_df = onehot(onehot_df, yearbin_df, "YearRemodAddBin", None)
       return onehot_df

if __name__=='__main__':
    print('start here!!!')
    start_time = time.time()
    ttt_data_frame = pd.DataFrame(
                         {'Id': [1, 2, 3], 
                          'SalePrice': ['a', 'b', 'c']
                         }, 
                         index=None)
    train_data_frame = pd.read_csv('./data/train.csv')
    test_data_frame = pd.read_csv('./data/test.csv')
    

    train_cols = list(set(train_data_frame.columns)-set(['Id', 'SalePrice']))
    df = train_data_frame.info()
#    df = train_data_frame.describe()
    print(df)
    miss = train_data_frame.isnull().sum()/len(train_data_frame)
    miss = miss[miss > 0]
    miss.sort(inplace=True)
    print(miss)
    miss = miss.to_frame()
    miss.columns = ['count']
    miss.index.names = ['Name']
    miss['Name'] = miss.index
    
    target = np.log(train_data_frame['SalePrice'])
    print('The skewness of SalePrice is {0}, kurtosis is {1}'.format(
           train_data_frame['SalePrice'].skew(),
           train_data_frame['SalePrice'].kurtosis()))
    print ('Skewness is {0} kurtosis is {1}'.format(target.skew(), 
                                                    target.kurtosis()))
#    sns.distplot(target)
    
    numeric_data = train_data_frame.select_dtypes(include=[np.number])
    cat_data = train_data_frame.select_dtypes(exclude=[np.number])
    print("There are {} numeric and {} categorical columns in train data".format(
           numeric_data.shape[1], cat_data.shape[1]))
    
    corr = numeric_data.corr()
#    sns.heatmap(corr)
    corr_SalePrice = corr['SalePrice'].copy()
    corr_SalePrice.sort(ascending=False)
    print('variance of columns is ', numeric_data.var())
    print('type of corr_SalePrice is ', type(corr_SalePrice))
    
    print(corr_SalePrice[:15], '\n') #top 15 values
    print('----------------------')
    print(corr_SalePrice[-5:])   #last 5 values
    print(train_data_frame['OverallQual'].unique())
    pivot = train_data_frame.pivot_table(index='OverallQual', values='SalePrice', 
                              aggfunc=np.median)
    pivot = pivot.copy()
    pivot.sort()
#    pivot.plot(kind='bar', color='red')
    
    sns.jointplot(x=train_data_frame['GrLivArea'], 
                  y=train_data_frame['SalePrice'])
    
    sp_pivot = train_data_frame.pivot_table(index='SaleCondition', 
                                            values='SalePrice', 
                                            aggfunc=np.median)
    sp_pivot = sp_pivot.copy()
    sp_pivot.sort()
    
    print(sp_pivot)
#    sp_pivot.plot(kind='bar',color='red')
    
    cat = [f for f in train_data_frame.columns if 
             train_data_frame.dtypes[f] == 'object']


    cat_data['SalePrice'] = train_data_frame.SalePrice.values
    k = anova(cat_data, cat) 
    k['disparity'] = np.log(1./k['pval'].values) 
    sns.barplot(data=k, x = 'features', y='disparity') 
    plt.xticks(rotation=90)
    
    num = [f for f in train_data_frame.columns if 
             train_data_frame.dtypes[f] != 'object']
    num.remove('Id')
    nd = pd.melt(train_data_frame, value_vars = num)
#    n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
#    n1 = n1.map(sns.distplot, 'value')
    
    
    cat = [f for f in train_data_frame.columns if 
              train_data_frame.dtypes[f] == 'object']
              
    train_data_frame.drop(train_data_frame[train_data_frame['GrLivArea'] > 4000].index, 
                          inplace=True)
    
    test_data_frame.loc[666, 'GarageQual'] = "TA"
    test_data_frame.loc[666, 'GarageCond'] = "TA"
    test_data_frame.loc[666, 'GarageFinish'] = "Unf"
    test_data_frame.loc[666, 'GarageYrBlt'] = "1980"
    
    test_data_frame.loc[1116, 'GarageType'] = np.nan

    alldata = train_data_frame.append(test_data_frame)
    train = train_data_frame
    test = test_data_frame
    lot_frontage_by_neighborhood = train['LotFrontage'].groupby(train['Neighborhood'])
    for key, group in lot_frontage_by_neighborhood:
        idx = (alldata['Neighborhood'] == key) & (alldata['LotFrontage'].isnull())
        alldata.loc[idx, 'LotFrontage'] = group.median()
    
    alldata["MasVnrArea"].fillna(0, inplace=True)
    alldata["BsmtFinSF1"].fillna(0, inplace=True)
    alldata["BsmtFinSF2"].fillna(0, inplace=True)
    alldata["BsmtUnfSF"].fillna(0, inplace=True)
    alldata["TotalBsmtSF"].fillna(0, inplace=True)
    alldata["GarageArea"].fillna(0, inplace=True)
    alldata["BsmtFullBath"].fillna(0, inplace=True)
    alldata["BsmtHalfBath"].fillna(0, inplace=True)
    alldata["GarageCars"].fillna(0, inplace=True)
    alldata["GarageYrBlt"].fillna(0.0, inplace=True)
    alldata["PoolArea"].fillna(0, inplace=True)
    
    qual_dict = {np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    name = np.array(['ExterQual','PoolQC' ,'ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu', 'GarageQual','GarageCond'])
    
    for i in name:
         alldata[i] = alldata[i].map(qual_dict).astype(int)
    
    alldata["BsmtExposure"] = alldata["BsmtExposure"].map({np.nan: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)
    
    bsmt_fin_dict = {np.nan: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    alldata["BsmtFinType1"] = alldata["BsmtFinType1"].map(bsmt_fin_dict).astype(int)
    alldata["BsmtFinType2"] = alldata["BsmtFinType2"].map(bsmt_fin_dict).astype(int)
    alldata["Functional"] = alldata["Functional"].map({np.nan: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)
    
    alldata["GarageFinish"] = alldata["GarageFinish"].map({np.nan: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)
    alldata["Fence"] = alldata["Fence"].map({np.nan: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)
    
    #encoding data
    alldata["CentralAir"] = (alldata["CentralAir"] == "Y") * 1.0
    varst = np.array(['MSSubClass','LotConfig','Neighborhood','Condition1','BldgType','HouseStyle','RoofStyle','Foundation','SaleCondition'])
    
    for x in varst:
        factorize(alldata, x)
    
    #encode variables and impute missing values
    alldata = factorize(alldata, "MSZoning", "RL")
    alldata = factorize(alldata, "Exterior1st", "Other")
    alldata = factorize(alldata, "Exterior2nd", "Other")
    alldata = factorize(alldata, "MasVnrType", "None")
    alldata = factorize(alldata, "SaleType", "Oth")
    
    alldata["IsRegularLotShape"] = (alldata["LotShape"] == "Reg") * 1
    alldata["IsLandLevel"] = (alldata["LandContour"] == "Lvl") * 1
    alldata["IsLandSlopeGentle"] = (alldata["LandSlope"] == "Gtl") * 1
    alldata["IsElectricalSBrkr"] = (alldata["Electrical"] == "SBrkr") * 1
    alldata["IsGarageDetached"] = (alldata["GarageType"] == "Detchd") * 1
    alldata["IsPavedDrive"] = (alldata["PavedDrive"] == "Y") * 1
    alldata["HasShed"] = (alldata["MiscFeature"] == "Shed") * 1
    alldata["Remodeled"] = (alldata["YearRemodAdd"] != alldata["YearBuilt"]) * 1
    
    #Did the modeling happened during the sale year
    alldata["RecentRemodel"] = (alldata["YearRemodAdd"] == alldata["YrSold"]) * 1
    
    # Was this house sold in the year it was built?
    alldata["VeryNewHouse"] = (alldata["YearBuilt"] == alldata["YrSold"]) * 1
    alldata["Has2ndFloor"] = (alldata["2ndFlrSF"] == 0) * 1
    alldata["HasMasVnr"] = (alldata["MasVnrArea"] == 0) * 1
    alldata["HasWoodDeck"] = (alldata["WoodDeckSF"] == 0) * 1
    alldata["HasOpenPorch"] = (alldata["OpenPorchSF"] == 0) * 1
    alldata["HasEnclosedPorch"] = (alldata["EnclosedPorch"] == 0) * 1
    alldata["Has3SsnPorch"] = (alldata["3SsnPorch"] == 0) * 1
    alldata["HasScreenPorch"] = (alldata["ScreenPorch"] == 0) * 1
    
    #setting levels with high count as 1 and rest as 0
    #you can check for them using value_counts function
    alldata["HighSeason"] = alldata["MoSold"].replace({1: 0, 2: 0, 3: 0, 4: 1, 
                                                      5: 1, 6: 1, 7: 1, 8: 0, 
                                                      9: 0, 10: 0, 11: 0, 12: 0})
    alldata["NewerDwelling"] = alldata["MSSubClass"].replace({
                                  20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 
                                  70: 0, 75: 0, 80: 0, 85: 0, 90: 0, 120: 1, 
                                  150: 0, 160: 0, 180: 0, 190: 0})

    print('alldata.shape is ', alldata.shape)
    
    alldata2 = train.append(test)

    alldata["SaleCondition_PriceDown"] = alldata2.SaleCondition.replace({
                                   'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 
                                   'Family': 1, 'Normal': 0, 'Partial': 0})

    # house completed before sale or not
    alldata["BoughtOffPlan"] = alldata2.SaleCondition.replace({
                                  "Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                  "Family" : 0, "Normal" : 0, "Partial" : 1})
    alldata["BadHeating"] = alldata2.HeatingQC.replace({
                                 'Ex': 0, 'Gd': 0, 
                                 'TA': 0, 'Fa': 1, 'Po': 1})

    area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 
                 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 
                 '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
                 'LowQualFinSF', 'PoolArea' ]

    alldata["TotalArea"] = alldata[area_cols].sum(axis=1)
    alldata["TotalArea1st2nd"] = alldata["1stFlrSF"] + alldata["2ndFlrSF"]
    alldata["Age"] = 2010 - alldata["YearBuilt"]
    alldata["TimeSinceSold"] = 2010 - alldata["YrSold"]
    alldata["SeasonSold"] = alldata["MoSold"].map({12:0, 1:0, 2:0, 3:1, 4:1, 
                                                   5:1, 6:2, 7:2, 8:2, 9:3, 
                                                   10:3, 11:3}).astype(int)
    alldata["YearsSinceRemodel"] = alldata["YrSold"] - alldata["YearRemodAdd"]
    
    # Simplifications of existing features into bad/average/good based on counts
    alldata["SimplOverallQual"] = alldata.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
    alldata["SimplOverallCond"] = alldata.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
    alldata["SimplPoolQC"] = alldata.PoolQC.replace({1 : 1, 2 : 1, 3 : 2, 4 : 2})
    alldata["SimplGarageCond"] = alldata.GarageCond.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    alldata["SimplGarageQual"] = alldata.GarageQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    alldata["SimplFireplaceQu"] = alldata.FireplaceQu.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    alldata["SimplFireplaceQu"] = alldata.FireplaceQu.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    alldata["SimplFunctional"] = alldata.Functional.replace({1 : 1, 2 : 1, 3 : 2, 4 : 2, 5 : 3, 6 : 3, 7 : 3, 8 : 4})
    alldata["SimplKitchenQual"] = alldata.KitchenQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    alldata["SimplHeatingQC"] = alldata.HeatingQC.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    alldata["SimplBsmtFinType1"] = alldata.BsmtFinType1.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})
    alldata["SimplBsmtFinType2"] = alldata.BsmtFinType2.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})
    alldata["SimplBsmtCond"] = alldata.BsmtCond.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    alldata["SimplBsmtQual"] = alldata.BsmtQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    alldata["SimplExterCond"] = alldata.ExterCond.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    alldata["SimplExterQual"] = alldata.ExterQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})
    
    #grouping neighborhood variable based on this plot
    train['SalePrice'].groupby(train['Neighborhood']).median().sort_values().plot(kind='bar')
    print('get here 222!')
    
    neighborhood_map = {"MeadowV" : 0, "IDOTRR" : 1, "BrDale" : 1, 
                        "OldTown" : 1, "Edwards" : 1, "BrkSide" : 1,
                        "Sawyer" : 1, "Blueste" : 1, "SWISU" : 2, 
                        "NAmes" : 2, "NPkVill" : 2, "Mitchel" : 2, 
                        "SawyerW" : 2, "Gilbert" : 2, "NWAmes" : 2, 
                        "Blmngtn" : 2, "CollgCr" : 2, "ClearCr" : 3, 
                        "Crawfor" : 3, "Veenker" : 3, "Somerst" : 3, 
                        "Timber" : 3, "StoneBr" : 4, "NoRidge" : 4, 
                        "NridgHt" : 4}

    alldata['NeighborhoodBin'] = alldata2['Neighborhood'].map(neighborhood_map)
    alldata.loc[alldata2.Neighborhood == 'NridgHt', "Neighborhood_Good"] = 1
    alldata.loc[alldata2.Neighborhood == 'Crawfor', "Neighborhood_Good"] = 1
    alldata.loc[alldata2.Neighborhood == 'StoneBr', "Neighborhood_Good"] = 1
    alldata.loc[alldata2.Neighborhood == 'Somerst', "Neighborhood_Good"] = 1
    alldata.loc[alldata2.Neighborhood == 'NoRidge', "Neighborhood_Good"] = 1
    alldata["Neighborhood_Good"].fillna(0, inplace=True)
    alldata["SaleCondition_PriceDown"] = alldata2.SaleCondition.replace({'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})
    
    # House completed before sale or not
    alldata["BoughtOffPlan"] = alldata2.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})
    alldata["BadHeating"] = alldata2.HeatingQC.replace({'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})
    
    train_new = alldata[alldata['SalePrice'].notnull()]
    test_new = alldata[alldata['SalePrice'].isnull()]
                       
    numeric_features = [f for f in train_new.columns if train_new[f].dtype != object]
    skewed = train_new[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
    skewed = skewed[skewed > 0.75]
    skewed = skewed.index
    train_new[skewed] = np.log1p(train_new[skewed])
    test_new[skewed] = np.log1p(test_new[skewed])
    del test_new['SalePrice']

    scaler = StandardScaler()
    scaler.fit(train_new[numeric_features])
    scaled = scaler.transform(train_new[numeric_features])
    
    for i, col in enumerate(numeric_features):
        train_new[col] = scaled[:,i]
    
    numeric_features.remove('SalePrice')
    scaled = scaler.fit_transform(test_new[numeric_features])
    
    for i, col in enumerate(numeric_features):
        test_new[col] = scaled[:,i]

    onehot_df = munge_onehot(train)

    neighborhood_train = pd.DataFrame(index=train_new.shape)
    neighborhood_train['NeighborhoodBin'] = train_new['NeighborhoodBin']
    neighborhood_test = pd.DataFrame(index=test_new.shape)
    neighborhood_test['NeighborhoodBin'] = test_new['NeighborhoodBin']
    
    onehot_df = onehot(onehot_df, neighborhood_train, 'NeighborhoodBin', None)
    
    train_new = train_new.join(onehot_df) 
    
    onehot_df_te = munge_onehot(test)
    onehot_df_te = onehot(onehot_df_te, neighborhood_test, "NeighborhoodBin", None)
    test_new = test_new.join(onehot_df_te)
    print('test_new.shape is ', test_new.shape)
    
    drop_cols = ["_Exterior1st_ImStucc", "_Exterior1st_Stone",
                 "_Exterior2nd_Other","_HouseStyle_2.5Fin",
                 "_RoofMatl_Membran", "_RoofMatl_Metal", 
                 "_RoofMatl_Roll", "_Condition2_RRAe", 
                 "_Condition2_RRAn", "_Condition2_RRNn", 
                 "_Heating_Floor", "_Heating_OthW", 
                 "_Electrical_Mix", "_MiscFeature_TenC", 
                 "_GarageQual_Ex",  "_PoolQC_Fa"]
    train_new.drop(drop_cols, axis=1, inplace=True)
    print('train_new.shape is ', train_new.shape)
    
    test_new.drop(["_MSSubClass_150"], axis=1, inplace=True)

    # Drop these columns
    drop_cols = ["_Condition2_PosN", "_MSZoning_C (all)", "_MSSubClass_160"]
    
    train_new.drop(drop_cols, axis=1, inplace=True)
    test_new.drop(drop_cols, axis=1, inplace=True)
    
    label_df = pd.DataFrame(index=train_new.index, columns=['SalePrice'])
    label_df['SalePrice'] = np.log(train['SalePrice'])
    print("Training set size:", train_new.shape)
    print("Test set size:", test_new.shape)
    
    print('start xgboost learning ...')
    regr = xgb.XGBRegressor(colsample_bytree=0.2,
                       gamma=0.0,
                       learning_rate=0.05,
                       max_depth=6,
                       min_child_weight=1.5,
                       n_estimators=7200,
                       reg_alpha=0.9,
                       reg_lambda=0.6,
                       subsample=0.2,
                       seed=42,
                       silent=1)

    
    train_new_one = pd.get_dummies(train_new)
    test_new_one = pd.get_dummies(test_new)
    
    print('shape of train_new_one is ', train_new_one.shape)
    print('shape of test_new_one is ', test_new)
    print('diff column name is ', 
          set(train_new.columns)-set(test_new_one.columns))
    
#    regr.fit(train_new_one, label_df)
#    y_pred = regr.predict(train_new_one)
#    y_test = label_df
#    print("XGBoost score on training set: ", rmse(y_test, y_pred))
#    
#
#    # make prediction on test set.
#    y_pred_xgb = regr.predict(test_new_one)
#    
#    #submit this prediction and get the score
#    pred1 = pd.DataFrame({'Id': test['Id'], 'SalePrice': np.exp(y_pred_xgb)})
#    pred1.to_csv('xgbnono.csv', header=True, index=False)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    print(train_data_frame.columns[train_data_frame.isnull().any()])
    
    