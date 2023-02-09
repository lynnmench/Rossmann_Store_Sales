#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Lynn Menchaca

Date: 07Feb2023

Project: Rossmann Stores Sales Forcasting

Resources:
    "The Ultimate Guide to 12 Dimensionality Reduction Techniques (with Python codes)" by Pulkit Sharam
    https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/
    
    "Feature Selection Techniques in Machine Learning" by Aman Gupta
    https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/
    
    
    

"""

"""

Feature Selection Methods:
    - Uivariate Selection
    - SelectKBest Algorithm
    - Feature Importance
    - Pearson Correlation Coefficient
    - Information Gain

Models:
    - asdf

"""




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import time


#from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
#from sklearn.ensemble import ExtraTreesClassifier
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


#reading in the Rossmann Store data set
# cleaned files - is post feature engineering
# shapped files - the features have been transformed and scaled

data_file_path = '/Users/lynnpowell/Documents/DS_Projects/Data_Files/Rossmann_Store_Sales_Data/'

train = pd.read_csv(data_file_path+'cleaned_train.csv')
test = pd.read_csv(data_file_path+'cleaned_test.csv')
train_shape = pd.read_csv(data_file_path+'shaped_train.csv')
test_shape = pd.read_csv(data_file_path+'shaped_test.csv')

#X = train.drop('Sales', axis=1)
#y = train['Sales']

#X = train_shape.drop('Sales', axis=1)
#y = train_shape['Sales']

# --------------------- Feature Selection Methods ---------------------


#### Pearson Correlation ####
# pearson correlation requires data to be normally distributed
# however results for this project are the exact same
# the closer to 1 the more correlation between features

#df_corr = train.corr()
df_corr = train_shape.corr()
df_corr = train.corr()
target_corr = df_corr['Sales'].abs().sort_values(ascending=False)
target_bin_corr = target_corr.drop(labels=(['Sales']))
corr_feat = target_bin_corr[target_bin_corr > 0.1].index.values.tolist()
corr_feat_df = pd.DataFrame(data=corr_feat, columns=['Features'])
corr_order_feat = target_bin_corr.index.values.tolist()
corr_feat_full_df = pd.DataFrame(data=corr_order_feat, columns=['Features']).reset_index(drop=True)


#### Information Gain ####
#Looking to see what highly correlated features are important to the final answer

#X = train.drop('Sales', axis=1)
#y = train['Sales']

X = train_shape.drop('Sales', axis=1)
y = train_shape['Sales']

mutual_info_values = mutual_info_classif(X,y)
mutual_info = pd.Series(mutual_info_values, index=X.columns)
mutual_info.sort_values(ascending=False)
mutual_info_df = mutual_info.sort_values(ascending=False).to_frame().reset_index()
mutual_info_df.columns=['Features','Mutual Info']

##### Feature Importance: #####
# The data does not need to be scaled
#This method provides a score for each feature of your data frame
#The higher the score the more relevant the data is

#### Data set is to big for this method on my computer ####

#X = train.drop('Sales', axis=1)
#y = train['Sales']

#model = ExtraTreesClassifier()
#etc = model.fit(X,y)

#print(etc.feature_importances_)
#feat_impotant = pd.Series(etc.feature_importances_, index=X.columns)
#feat_impotant.nlargest(20).plot(kind='barh')
#feat_impot_df = feat_impotant.sort_values(ascending=False).to_frame().reset_index()
#feat_impot_df.columns=['Features','ETC Score']


##### Fisher's Score: #####
# Most popular supervised feature selection method
X = train_shape.drop('Sales', axis=1)
y = train_shape['Sales']
idx = fisher_score.fisher_score(X, y, mode='rank')


fisher = fisher_score.fisher_score(X,y)
f_score = pd.Series(fisher, index=X.columns)
f_score_df = f_score.sort_values(ascending=False).to_frame().reset_index()
f_score_df.columns=['Features','Fisher Score']


### Apply SelectKBest Algorithm
### Also refered to as information gain?

X = train_shape.drop('Sales', axis=1)
y = train_shape['Sales']
X_col = X.shape[1]

ordered_rank_features = SelectKBest(score_func=chi2, k=2)
ordered_feature = ordered_rank_features.fit(X,y)

univar_score = pd.DataFrame(ordered_feature.scores_, columns=['Score'])
univar_col = pd.DataFrame(X.columns)

univar_df = pd.concat([univar_col, univar_score], axis=1)
univar_df.columns=['Features','Score']

# For SelectKBest Algorithm the higher the score the higher the feature importance
univar_df['Score'].sort_values(ascending=False)
univar_df = univar_df.nlargest(50, 'Score').reset_index(drop=True)


