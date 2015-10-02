# -*- coding: utf-8 -*-
"""
Created on Wed Aug 05 23:47:00 2015

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
data.dtypes
data.head(20)

nrow = data.shape[0]
ncol = data.shape[1]

data['Survived'].value_counts()

Sex_Sur = pd.crosstab(data['Sex'], data['Survived'])
Sex_Sur_norm = Sex_Sur.div(Sex_Sur.sum(1), axis = 0)
Sex_Sur_norm.plot(kind = 'barh', stacked = True)

Pclass_Sur = pd.crosstab(data['Pclass'], data['Survived'])
Pclass_Sur_norm = Pclass_Sur.div(Pclass_Sur.sum(1), axis = 0)
Pclass_Sur_norm.plot(kind = 'barh', stacked = True)

SibSp_Sur = pd.crosstab(data['SibSp'], data['Survived'])
SibSp_Sur_norm = SibSp_Sur.div(SibSp_Sur.sum(1), axis = 0)
SibSp_Sur_norm.plot(kind = 'barh', stacked = True)

Parch_Sur = pd.crosstab(data['Parch'], data['Survived'])
Parch_Sur_norm = Parch_Sur.div(Parch_Sur.sum(1), axis = 0)
Parch_Sur_norm.plot(kind = 'barh', stacked = True)

Embarked_Sur = pd.crosstab(data['Embarked'], data['Survived'])
Embarked_Sur_norm = Embarked_Sur.div(Embarked_Sur.sum(1), axis = 0)
Embarked_Sur_norm.plot(kind = 'barh', stacked = True)

data.boxplot(['Age'], by = ['Survived'])

######################## manage Passenger Class ###################
# Passenger Class - 1
Pclass_dum = data['Pclass'].copy()
Pclass_dum[data['Pclass'] != 1] = 0
data['Pclass_1'] = Pclass_dum

# Passenger Class - 2
Pclass_dum = data['Pclass'].apply(lambda x: 1 if x == 2 else 0)
data['Pclass_2'] = Pclass_dum

# Passenger Class - 3
Pclass_dum = data['Pclass'].apply(lambda x: 1 if x == 3 else 0)
data['Pclass_3'] = Pclass_dum

data.drop(['Pclass'], axis = 1)

############################# manage Sex ###########################
data['Sex_male'] = np.where(data['Sex'] == 'male', 1, 0)
data['Sex_female'] = np.where(data['Sex'] == 'female', 1, 0)
data.drop(['Sex'], axis = 1)

############################# manage Embarked ###########################
data['Embarked_S'] = np.where(data['Embarked'] == 'S', 1, 0)
data['Embarked_Q'] = np.where(data['Embarked'] == 'Q', 1, 0)
data['Embarked_C'] = np.where(data['Embarked'] == 'C', 1, 0)

data.drop(['Embarked'], axis = 1)

############################# manage Age ###########################
data['Age_fill'] = data['Age'].fillna(np.mean(data['Age']))


# create train data set 
data.dtypes
data.head()
xTrain_1 = data.loc[:, ['SibSp', 'Parch', 'Fare']]
xTrain_2 = data.iloc[:, 11:]
xTrain = pd.concat([xTrain_1, xTrain_2], axis = 1)

yTrain = data.loc[:, 'Survived']

ncol_train = xTrain.shape[1]

# parameter tuning
feat_num = range(3, ncol_train + 1)
tree_num = [100, 500, 1000]
clf = GridSearchCV(estimator = RandomForestClassifier(oob_score = False, max_features = feat_num, 
                                                      max_depth=None, min_samples_split=1), 
                   param_grid = dict(n_estimators = tree_num, max_features = feat_num), cv = 5, scoring='roc_auc')
clf.fit(xTrain.values, yTrain.values)

clf.best_estimator_
clf.best_score_
clf.grid_scores_

# fit the model
model_rf = RandomForestClassifier(n_estimators = 1000, max_features = 9, 
                                  oob_score = False, 
                                  max_depth=None, min_samples_split = 1)
model_rf.fit(xTrain.values, yTrain.values)  
model_rf.predict(xTrain.values)             
# test the model                 
test_set = pd.read_csv('test.csv', header = 0, index_col = 0)

######################## manage Passenger Class ###################
# Passenger Class - 1
Pclass_dum = test_set['Pclass'].copy()
Pclass_dum[test_set['Pclass'] != 1] = 0
test_set['Pclass_1'] = Pclass_dum

# Passenger Class - 2
Pclass_dum = test_set['Pclass'].apply(lambda x: 1 if x == 2 else 0)
test_set['Pclass_2'] = Pclass_dum

# Passenger Class - 3
Pclass_dum = test_set['Pclass'].apply(lambda x: 1 if x == 3 else 0)
test_set['Pclass_3'] = Pclass_dum

test_set.drop(['Pclass'], axis = 1)

############################# manage Sex ###########################
test_set['Sex_male'] = np.where(test_set['Sex'] == 'male', 1, 0)
test_set['Sex_female'] = np.where(test_set['Sex'] == 'female', 1, 0)
test_set.drop(['Sex'], axis = 1)

############################# manage Embarked ###########################
test_set['Embarked_S'] = np.where(test_set['Embarked'] == 'S', 1, 0)
test_set['Embarked_Q'] = np.where(test_set['Embarked'] == 'Q', 1, 0)
test_set['Embarked_C'] = np.where(test_set['Embarked'] == 'C', 1, 0)

data.drop(['Embarked'], axis = 1)

############################# manage Age ###########################
test_set['Age_fill'] = test_set['Age'].fillna(np.mean(test_set['Age']))


# create train data set 
test_set.dtypes
test_set.head()
xTest_1 = test_set.loc[:, ['SibSp', 'Parch', 'Fare']]
xTest_2 = test_set.iloc[:, 10:]
xTest = pd.concat([xTest_1, xTest_2], axis = 1)
nrow_test = xTest.shape[0]

yTest = pd.read_csv('gendermodel.csv', header = 0, index_col = 0)
###
xTest.isnull().values.sum()
xTest.isnull().sum()
xTest['Fare'] = xTest['Fare'].fillna(np.mean(test_set['Fare']))
###
yTest = np.ravel(yTest.values)
xTest = xTest.values

###
mean_List = []
std_List = []
for i in range(0, 10):
    CV = KFold(nrow_test, n_folds = 10, shuffle = True)
    thisScore = cross_val_score(model_rf, xTest, yTest, cv=CV, scoring='roc_auc')
    mean_List.append(np.mean(thisScore))
    std_List.append(np.std(thisScore))
    
###
y_pred = model_rf.predict(xTest)
y_pred_lt = y_pred.tolist()
out_pred = {'PassengerId':range(892, 1310), 'Survived':y_pred_lt}
out_pred = pd.DataFrame(out_pred)
out_pred.to_csv('test2.csv', header = True, index = False)