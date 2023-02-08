#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Lynn Menchaca

Date: 07Feb2023

Project: Rossmann Stores Sales Forcasting

Resources:
    "The Ultimate Guide to 12 Dimensionality Reduction Techniques (with Python codes)" by Pulkit Sharam
    https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/

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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier