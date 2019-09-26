# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 23:43:16 2019

@author: Manoochehr
"""
import numpy as np

def preprocessFunc(train_data,test_data):
    train_isnull = train_data.isnull().sum()
    print(train_isnull[train_isnull > 0])
    test_isnull = test_data.isnull().sum()
    print(test_isnull[test_isnull > 0])
    
    # Drop nan columns with more than 60% nan value
    features_toBeAbandoned = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
    train_data.drop(features_toBeAbandoned, axis=1, inplace=True)
    test_data.drop(features_toBeAbandoned, axis=1, inplace=True)
    
    # filling other Missing values
    train_data['LotFrontage'] = train_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    test_data['LotFrontage'] = test_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    
    train_data['Electrical'] = train_data['Electrical'].fillna(train_data['Electrical'].mode()[0])
    train_data['MasVnrType'] = train_data['MasVnrType'].fillna('None')
    train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].mode()[0])
    test_data['MasVnrType'] = test_data['MasVnrType'].fillna('None')
    test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(test_data['MasVnrArea'].mode()[0])
    
    for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
        train_data[col] = train_data[col].fillna('None')
        test_data[col] = test_data[col].fillna('None')
        
    for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
        test_data[col] = test_data[col].fillna(test_data[col].mode()[0])
        
    
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        train_data[col] = train_data[col].fillna('None')
        test_data[col] = test_data[col].fillna('None')
    
    for col in ['GarageCars','GarageArea','GarageYrBlt','1stFlrSF','TotRmsAbvGrd','KitchenAbvGr']:
        train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
        test_data[col] = test_data[col].fillna(test_data[col].mode()[0])
    
    for col in ['MSZoning','Utilities','KitchenQual','Functional','SaleType']:
        test_data[col] = test_data[col].fillna(test_data[col].mode()[0])
    
    train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna('None')
    test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna('None')
    
    test_data['Exterior1st'] = test_data['Exterior1st'].fillna(test_data['Exterior1st'].mode()[0])
    test_data['Exterior2nd'] = test_data['Exterior2nd'].fillna(test_data['Exterior1st'].mode()[0])
    
    train_isnull2 = train_data.isnull().sum()
    print(train_isnull2[train_isnull2 > 0])
    test_isnull2 = test_data.isnull().sum()
    print(test_isnull2[test_isnull2 > 0])

    return train_data,test_data

def AddPolyFeature(Input):
    Input["MasVnrArea-s2"] = Input["MasVnrArea"] ** 2
    Input["BsmtFinSF1-2"] = Input["BsmtFinSF1"] ** 2
    Input["BsmtFinSF2-2"] = Input["BsmtFinSF2"] ** 2
    Input["BsmtUnfSF-2"] = Input["BsmtUnfSF"] ** 2
    Input["TotalBsmtSF-s2"] = Input["TotalBsmtSF"] ** 2
    Input["2ndFlrSF-2"] = Input["2ndFlrSF"] ** 2
    Input["LowQualFinSF-2"] = Input["LowQualFinSF"] ** 2
    Input["WoodDeckSF-2"] = Input["WoodDeckSF"] ** 2
    Input["EnclosedPorch-2"] = Input["EnclosedPorch"] ** 2
    Input["3SsnPorch-2"] = Input["3SsnPorch"] ** 2
    Input["ScreenPorch-2"] = Input["ScreenPorch"] ** 2
    Input["YrSold-2"] = Input["YrSold"] ** 2    

    return Input
