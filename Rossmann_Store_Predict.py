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
    
    "7 Ways to Handel Large Data Files for Machine Learning" by Jason Brownlee
    https://machinelearningmastery.com/large-data-files-machine-learning/
    
    
    
    
    

"""

"""
The purpose of this project is to continue gaining an understanding of feature
selection methods and experiment with different machine learning models. 


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
import random
import math
import statistics



from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
#from sklearn.ensemble import ExtraTreesClassifier
#from skfeature.function.similarity_based import fisher_score

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_squared_error


#reading in the Rossmann Store data set
# cleaned files - is post feature engineering
# shapped files - the features have been transformed and scaled

data_file_path = '/Users/lynnpowell/Documents/DS_Projects/Data_Files/Rossmann_Store_Sales_Data/'

train = pd.read_csv(data_file_path+'cleaned_train.csv')
test = pd.read_csv(data_file_path+'cleaned_test.csv')
#train_shape = pd.read_csv(data_file_path+'shaped_train.csv', index_col=False)
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

df_corr = train.corr()
#plt.figure(figsize=(10,6))
#sns.heatmap(df_corr, annot=True)
target_corr = df_corr['Sales'].abs().sort_values(ascending=False)
target_bin_corr = target_corr.drop(labels=(['Sales']))
corr_feat = target_bin_corr[target_bin_corr > 0.1].index.values.tolist()
corr_feat_df = pd.DataFrame(data=corr_feat, columns=['Features'])
corr_order_feat = target_bin_corr.index.values.tolist()
corr_feat_full_df = pd.DataFrame(data=corr_order_feat, columns=['Features']).reset_index(drop=True)

#List of features with high correlation to each other
train_lst = train.columns.tolist()
train_lst.remove('Sales')
train_corr = train[train_lst].corr()
feat_high_corr_lst = []
for feat in train_lst:
    feat_corr = train_corr[feat].abs().sort_values(ascending=False)
    feat_bin_corr = feat_corr.drop(labels=([feat]))
    corr_feat = feat_bin_corr[feat_bin_corr > 0.5].index.values.tolist()
    for item in corr_feat:
        if item in feat_high_corr_lst:
            continue
        else:
            feat_high_corr_lst.append(item)

compare_feat_corr = train[feat_high_corr_lst].corr()

#### Coefficient ####
# The higher the coeffiencint values the better fit for the model
X = train_shape.drop('Sales', axis=1)
y = train_shape['Sales']

linr = LinearRegression()
linr.fit(X, y)
coefficients = linr.coef_
spend_coeff_bins = abs(pd.Series(coefficients,
                               index=X.columns)).sort_values(ascending=False)
#spend_coeff_bins = spend_coeff_bins[spend_coeff_bins > 0.9].index.values.tolist()

"""
Since this data set is so big the other feature selection methods stuggle to perform.
Based off the correlation values and coeffiecient values below are the features drops.
Features are dropped due to low impact to the Sales feature or high correlation to the other features.

StateHoliday: Test only has 0 or a
- also it doesn't matter what the holiday is lump a,b,c toghether as not 0

Calendar Feat Drop:  WeekOfYear, Day, Year(?)

promo drop: Promo2Since (?) or Promo2StartMonth(?) or both(?)


"""

drop_feat = ['StateHoliday_a', 'StateHoliday_b','WeekOfYear','Day']
train_shape.drop(drop_feat, axis=1, inplace=True)
train.drop(drop_feat, axis=1, inplace=True)
#test_shape.drop(drop_feat, axis=1, inplace=True)


#stores with the most entries in the data set
store_count = train['Store'].value_counts().sort_values(ascending=False)
sales_lst = train.groupby(['Store'])['Sales'].sum().sort_values(ascending=False)
sales_lst_values = sales_lst.index.values.tolist()
#total numberr of stores 1115

#taking the stores with the most sales and the least amount of sales
n_long = 410
n_short = 230
store_lst_long = sales_lst_values[:n_long] + sales_lst_values[-n_long:]
store_lst_short = sales_lst_values[:n_short] + sales_lst_values[-n_short:]


#### Information Gain ####
#Looking to see what highly correlated features are important to the final answer
#scale and normal distribution are for best results - information gain uses entropy
# this takes a long time to run using 19 features and a full data set.

X = train_shape.drop('Sales', axis=1)
y = train_shape['Sales']

"""
One idea was to create a loop to simulate cross validation method to evaluate
the information gain method on smaller data sets
Decided not to go this route becuase I don't want random rows becuase this just provided
select sales from stores
I want full sale sets from different stores.

#[3,17,47,73,97]
for rs in [3,7]:
    df_random = train_shape.sample(frac = 0.3, replace=False, random_state=rs)
    
    X = df_random.drop('Sales', axis=1)
    y = df_random['Sales']
    mutual_info_values = mutual_info_classif(X,y)
    mutual_info = pd.Series(mutual_info_values, index=X.columns)
    mutual_info_df = mutual_info.to_frame().reset_index()
    mutual_info_df.columns=['Features','Mutual Info_'+str(rs)]
    mutual_info_full = pd.merge(mutual_info_full,mutual_info_df,on='Features', how='left')
"""

mutual_info_values = mutual_info_classif(X,y)
mutual_info = pd.Series(mutual_info_values, index=X.columns)
mutual_info.sort_values(ascending=False)
mutual_info_df = mutual_info.sort_values(ascending=False).to_frame().reset_index()
mutual_info_df.columns=['Features','Mutual Info']

#export mutual information with full data frame
mutual_info_df.to_csv(data_file_path+'mutual_info_full_df.csv', index=False)


### Apply SelectKBest Algorithm
### Also refered to as information gain?
#Using the full data set this method "restarts the kernels..."

X = train_shape.loc[train['Store'].isin(store_lst_short)]
y = train_shape['Sales'].loc[train['Store'].isin(store_lst_short)]
X_col = X.shape[1]


ordered_rank_features = SelectKBest(score_func=chi2, k=X_col)
ordered_feature = ordered_rank_features.fit(X,y)

univar_score = pd.DataFrame(ordered_feature.scores_, columns=['Score'])
univar_col = pd.DataFrame(X.columns)

univar_df = pd.concat([univar_col, univar_score], axis=1)
univar_df.columns=['Features','Score']

#export SelectKBest and chi2 ranking with full feature list and select store data
univar_df.to_csv(data_file_path+'SelKBest_chi2_feat.csv', index=False)

# For SelectKBest Algorithm the higher the score the higher the feature importance
univar_df['Score'].sort_values(ascending=False)
univar_df = univar_df.nlargest(50, 'Score').reset_index(drop=True)


#### Forward Feature Selection ####
# using with linear regression ML model

X = train_shape.drop('Sales', axis=1)
y = train_shape['Sales']

linr = LinearRegression()
ffs = sfs(linr, k_features='best', forward=True, 
                                verbose=2, scoring='neg_mean_squared_error')
ffs.fit(X,y)
features = list(ffs.k_feature_names_)
ffs_feat = pd.DataFrame(features, columns=['Feature Ranking'])

#export Forward Feature Selection list made from full data frame
ffs_feat.to_csv(data_file_path+'FFS_Feature_Ranking.csv', index=False)

"""
Summary of feature selection using: 

high importance rank features:
    Store, Open, Promo, DayOfWeek, Assortment
    
Not sure if high or low importance:
    storeType_a,b,c , StateHoliday_null,0 , CompetitionDistance, CompetitionSince

Low importance features:
    Year, Month (important dates covered by state & school holiday), SchoolHoiday

Promo2, Promo2StartMonth, Promo2Since 
- all cover when promotion starts and how long it has been running for
- promo covers if a promotion is running that day going to just use this instead

"""

## Ways to imporove models with additional feature selection methods
"""
##### Need to use less features to run ######
# Data set is to big for this method on my computer, restarts kernels

##### Fisher's Score: #####
#Took to long with top 15 stores with most sales
# Most popular supervised feature selection method
X = train_shape.drop('Sales', axis=1)
y = train_shape['Sales']

#idx = fisher_score.fisher_score(X, y, mode='rank')
fisher = fisher_score.fisher_score(X,y)
f_score = pd.Series(fisher, index=X.columns)
f_score_df = f_score.sort_values(ascending=False).to_frame().reset_index()
f_score_df.columns=['Features','Fisher Score']


##### Feature Importance: #####
#This locked up and froze the computer used top 15 store with most sales
# The data does not need to be scaled
#This method provides a score for each feature of your data frame
#The higher the score the more relevant the data is

X = train.drop('Sales', axis=1)
y = train['Sales']

model = ExtraTreesClassifier()
etc = model.fit(X,y)

#print(etc.feature_importances_)
feat_impotant = pd.Series(etc.feature_importances_, index=X.columns)
#feat_impotant.nlargest(20).plot(kind='barh')
feat_impot_df = feat_impotant.sort_values(ascending=False).to_frame().reset_index()
feat_impot_df.columns=['Features','ETC Score']

#export original feature importance with full data frame
feat_impot_df.to_csv(data_file_path+'mutual_info_full_df.csv', index=False)
"""


#### ML Models #######


"""
# Evaluate models used for cross validation by splitting with different store IDs
# instead of by random rows in the full data set (traditional train test split)
# Doing this becuase there are 855 matching store IDs from the train and test data sets.


important_feat = ['Sales','Store','Open','Promo','DayOfWeek',
                  'Assortment','StoreType_a','StoreType_d','StateHoliday_0','CompetitionDistance']

train_shape_final = train_shape[important_feat]
train_final = train[important_feat]


def manual_model_eval(df, model):
    #full list of store values
    store_lst = df['Store'].unique()
    scores = []
    #best_estimators = {}
    seed_lst = [3,17,47,73,97]
    #seed_lst = [3,17]

    #For loop to simulate cross validation by store number
    for seed in seed_lst:
        #setting 70% to later use for train test split
        n = round(len(store_lst)*0.7)
        random.seed(seed)
        random_store_lst = random.sample(list(store_lst), k=n)
    
        #Generating Train - 70%, Test - 30% split
        X_train = df.loc[df['Store'].isin(random_store_lst)].drop('Sales', axis=1)
        X_test = df.loc[~df['Store'].isin(random_store_lst)].drop('Sales', axis=1)
        y_train = df['Sales'].loc[df['Store'].isin(random_store_lst)]
        y_test = df['Sales'].loc[~df['Store'].isin(random_store_lst)]
        
        start_time = time.time()
        model.fit(X_train,y_train)
        y_predict = model.predict(X_test)
        rmse_score = math.sqrt(mean_squared_error(y_test, y_predict))
        model_time = time.time() - start_time
        
        scores.append(rmse_score)
    
    return [statistics.median(scores), model_time]
        
    
#To keep track of model information
results = []

# Linear Regression
linr = LinearRegression()
linr_result = manual_model_eval(train_shape_final, linr)
results.append({
    'model':'Linear Regression',
    'Best_Score':linr_result[0],
    'Test_Time':linr_result[1]})

#Lasso Regression
lassor = Lasso(alpha=0.1)
lasso_result = manual_model_eval(train_shape_final, lassor)
results.append({
    'model':'Lasso_alpha_0.1',
    'Best_Score':lasso_result[0],
    'Test_Time':lasso_result[1]})

#Ridge Regression
ridger = Ridge(alpha=1)
ridge_result = manual_model_eval(train_shape_final, ridger)
results.append({
    'model':'Ridge_alpha_1',
    'Best_Score':ridge_result[0],
    'Test_Time':ridge_result[1]})

print(results)

#Comparing results to standard train, test split and cross validation
# using 5 fold and 70/30 train test split


model_eval_param = {
    'lasso': {
        'model': Lasso(),
        'params' : {
            'alpha': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]
        }
    },
    'ridge': {
        'model': Ridge(),
        'params' : {
            'alpha': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]
        }  
    }
}

X = train_shape_final.drop('Sales', axis=1)
y = train_shape_final['Sales']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)  

scores = []
best_parm = {}
#for loop for each model defined above
for name, ml in model_eval_param.items():
    model = ml['model']
    params = ml['params']
    clf = GridSearchCV(model, params, scoring='neg_root_mean_squared_error', cv=5, return_train_score=False)
    print(clf)
    clf.fit(X_train, y_train)
    scores.append({
        'model': name+'_cv_tranf',
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_parm[name] = clf.best_estimator_
    
print(scores)
"""

# Using the two different methods to evaluate the models the results are very close 
# Ridge Method: manual train test split results - 2548.6116, using sklearn library - 2599.552613
# Both results have the Ridge method just slightly better than the Lasso method

# For the rest of the models just going to use the sklearn library
# List of models evaluating:
    #Linear Regression, Lasso, Ridge, Decision Tree Regression, Random Forest, KNN
    #Support Vector Machine (SVM), Gausian, Polynomial

important_feat = ['Sales','Store','Open','Promo','DayOfWeek',
                  'Assortment','StoreType_a','StoreType_d','StateHoliday_0','CompetitionDistance']

train_shape_final = train_shape[important_feat]
train_final = train[important_feat]

def model_eval(df,model,params):
    X = df.drop('Sales', axis=1)
    y = df['Sales']
    
    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    
    if params == 'None':
        start_time = time.time()
        model.fit(X_train,y_train)
        y_predict = model.predict(X_test)
        rmse_score = math.sqrt(mean_squared_error(y_test, y_predict))
        model_time = time.time() - start_time
        
        return ['N/A', 'N/A', rmse_score, model_time]
        
    
    else:
        start_time = time.time()
        gs_cv = GridSearchCV(model, params, scoring='neg_root_mean_squared_error', cv=7, return_train_score=False)
        gs_cv.fit(X,y)
    
        #testing model with best parameters
        y_predict = gs_cv.predict(X_test)
        rmse_score = math.sqrt(mean_squared_error(y_test, y_predict))
        model_time = time.time() - start_time
    
        return [gs_cv.best_params_, gs_cv.best_score_, rmse_score, model_time]
    
results = []

# Linear Regression
linr = LinearRegression()
linr_result = model_eval(train_shape_final, linr, 'None')
results.append({
    'model':'Linear Regression',
    'Best_Params':linr_result[0],
    'CV_Score':linr_result[1],
    'ML_Score':linr_result[2],
    'Test_Time':linr_result[3]})

print(results)

rfc_hyperpar = {
    'criterion' : ['entropy', 'gini'],
    'max_depth' : [5, 10],
    'max_features' : ['log2', 'sqrt'],
    'min_samples_leaf' : [1,5],
    'min_samples_split' : [3,5],
    'n_estimators' : [6,9]
}
rfc = RandomForestClassifier(random_state=7)




