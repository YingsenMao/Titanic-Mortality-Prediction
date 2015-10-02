# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 16:18:32 2015

@author: Yingsen
"""

import os
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV


os.chdir('C:\\Users\\Yingsen\\Documents\\Kaggle\\titanic')

data = pd.read_csv('train.csv', header = 0, index_col = 0)

data.head()

###LabelEncoder() takes as input categorical values encoded as integers
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
pre_sex = le_sex.fit_transform(data.Sex)

pre_sex_2 = le_sex.fit(data.Sex)
test = ['male', 'male', 'male', 'female']
pre_sex_2.transform(test)
###Normalized Integer Labels
le_2 = preprocessing.LabelEncoder()
le_2.fit([1, 2, 2, 6])
le_2.classes_
le_2.transform([1, 1, 2, 6]) 
le_2.inverse_transform([0, 0, 1, 2])

###DictVectorizer expects data as a list of dictionaries, 
###where each dictionary is a data row with column names as keys
# first, convert the target column (categorical variable) into dictionary
from sklearn.feature_extraction import DictVectorizer

col_name = ['Pclass', 'Sex']
cat_df = data[col_name]
cat_dict = cat_df.T.to_dict().values()
vectorizer = DictVectorizer(sparse = False)
vec_x_cat_train = vectorizer.fit_transform( cat_dict )

cat_dict[:3]
vec_x_cat_train[:3, :]
vectorizer.get_feature_names()

cat_df.dtypes

col_name = ['Pclass', 'Sex']
cat_df_2 = data[col_name]
cat_df_2.dtypes
cat_df_2.loc[:, 'Pclass'] = cat_df_2.loc[:, 'Pclass'].astype(str)
cat_df_2.dtypes
cat_dict = cat_df.T.to_dict().values()
vectorizer = DictVectorizer(sparse = False)
vec_x_cat_train = vectorizer.fit_transform( cat_dict )

###Get Dummy Variables
pd.get_dummies(data['Sex'])
pd.get_dummies(data['Pclass'])
