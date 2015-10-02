# -*- coding: utf-8 -*-
"""
Created on Sat Aug 08 23:39:11 2015

@author: Yingsen
"""
import os
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import pylab as plot

os.chdir('C:\\Users\\Yingsen\\Documents\\Kaggle\\titanic')

data = pd.read_csv('train.csv', header = 0)

#data.info() 
#data.head()    

def cate_tran(data):
    """use Pclass, Sex, Embarked, SibSp, Parch, Fare, Age
    create the dummy variables for Pclass, Sex, and Embarked
    Normalize Fare
    For Age, fill missing value with mean and normalize it
    Divide Sibsp and Parch into 0(0) and more than 0(1), and normalize
    """
    Pclass_dum = pd.get_dummies(data.Pclass)
    Pclass_dum = Pclass_dum.rename(columns = lambda x: 'Pclass_' + str(x))
    Pclass_dum.drop(Pclass_dum.columns[-1], axis = 1, inplace = True)
    
    Sex_dum = pd.get_dummies(data.Sex)
    Sex_dum = Sex_dum.rename(columns = lambda x: 'Sex_' + str(x))
    Sex_dum.drop(Sex_dum.columns[-1], axis = 1, inplace = True)
    
    Embarked_dum = pd.get_dummies(data.Embarked)
    Embarked_dum = Embarked_dum.rename(columns = lambda x: 'Embarked_' + str(x))
    Embarked_dum.drop(Embarked_dum.columns[-1], axis = 1, inplace = True)
    
    SibSp_new = np.where(data.SibSp > 0, 1, 0)
    SibSp_new = SibSp_new.astype('float64')
    SibSp_new = pd.DataFrame(preprocessing.scale(SibSp_new), 
                             columns = ['SibSp_nor'],
                             index = Pclass_dum.index)

    Parch_new = np.where(data.Parch > 0, 1, 0)
    Parch_new = Parch_new.astype('float64')
    Parch_new = pd.DataFrame(preprocessing.scale(Parch_new), 
                             columns = ['Parch_nor'],
                             index = Pclass_dum.index)
    if data.Fare.isnull().any():
        data.Fare.fillna(np.mean(data.Fare), inplace = True)
    Fare_nor = pd.DataFrame(preprocessing.scale(data.Fare),
                            columns = ['Fare_nor'],
                            index = Pclass_dum.index)
            
    Age_nor = data.Age.fillna(np.mean(data.Age))
    Age_nor = pd.DataFrame(preprocessing.scale(Age_nor), 
                           columns = ['Age_nor'],
                           index = Pclass_dum.index)

    res = pd.concat([Pclass_dum, Sex_dum, Embarked_dum, SibSp_new, 
                        Parch_new, Fare_nor, Age_nor], axis = 1)
    return res


xTrain = cate_tran(data)
ncol_train = xTrain.shape[1]
nrow_train = xTrain.shape[0]
yTrain = data['Survived']

test_set = pd.read_csv('test.csv', header = 0)    
xTest = cate_tran(test_set)
xTest.info()

################
def get_score_train_test(cv_obj, x, y, model):
    train_score = []
    test_score = []
    for train_set, test_set in cv_obj:
        x_train = x.ix[train_set, :]
        y_train = y.ix[train_set]
        x_test = x.ix[test_set, :]
        y_test = y.ix[test_set]
        model.fit(x_train, y_train)
        train_score.append(model.score(x_train, y_train))
        test_score.append(model.score(x_test, y_test))
    return train_score, test_score

def variance_bias(x, y, model, fold_num = 10):
    nrow = x.shape[0]
    CV = KFold(nrow, fold_num, shuffle = True)
    train_lt, test_lt = get_score_train_test(CV, x, y, model)
    plot.figure()
    plot.plot(range(1, len(train_lt) + 1), train_lt, label = 'Train Score')
    plot.plot(range(1, len(test_lt) + 1), test_lt, label = 'Test Score')   
    plot.legend(loc = 'upper right')
    plot.show()

def get_acc(x, y, model, fold_num = 10):
    mean_List = []
    for i in range(0, 10):
        CV = KFold(nrow_train, n_folds = 5, shuffle = True)    
        thisScore = cross_val_score(model, x, y, cv=CV, scoring='accuracy')
        mean_List.append(np.mean(thisScore))
    return np.mean(mean_List)
################
    
################ Decision Tree ###############
model_DT = tree.DecisionTreeClassifier(criterion = 'entropy', 
                                  max_depth = None)
model_DT.fit(xTrain, yTrain)
model_DT.feature_importances_
pd.DataFrame(model_DT.feature_importances_, columns = ["Imp"], 
             index = xTrain.columns).sort(['Imp'], ascending = False)

model_DT.score(xTrain, yTrain)  #0.98204264870931535

dotfile = open('tree.dot', 'w')
tree.export_graphviz(model_DT, out_file = dotfile, 
                               feature_names = xTrain.columns)
dotfile.close()    

# in command line
# cd C:\Users\Yingsen\Documents\Kaggle\titanic  
# dot -Tpng tree3.dot -o tree3.png                                                    
#pred_matrix = metrics.confusion_matrix(model_DT.predict(xTrain), yTrain) 
#model_DT.score(xTrain, yTrain)
#float(539 + 178) / sum(sum(pred_matrix)) # same to above line

DT_acc = cross_val_score(model_DT, xTrain, yTrain, 
                         cv = 10, scoring = 'accuracy')
np.mean(DT_acc) 

#plot the variance and biae, seeking the best trade-off
variance_bias(xTrain, yTrain, model_DT)    
variance_bias(xTrain, yTrain, model_DT, fold_num = 5)  

model_RF = GridSearchCV(estimator = tree.DecisionTreeClassifier(criterion = 'entropy'), 
                   param_grid = dict(max_depth = range(1, 20)), 
                   cv = 5, scoring='accuracy')
model_RF.fit(xTrain, yTrain)

model_RF.best_estimator_ # max_depth = 4
model_RF.best_score_
model_RF.grid_scores_

model_DT_2 = tree.DecisionTreeClassifier(criterion = 'entropy', 
                                         max_depth = 4)
variance_bias(xTrain, yTrain, model_DT_2)    
variance_bias(xTrain, yTrain, model_DT_2, fold_num = 5) 
get_acc(xTrain, yTrain, model_DT_2) #0.80931579938484721
                        
mean_List = []
std_List = []
for i in range(0, 10):
    CV = KFold(nrow_train, n_folds = 10, shuffle = True)    
    thisScore = cross_val_score(model_DT_2, xTrain, yTrain, cv=CV, scoring='accuracy')
    mean_List.append(np.mean(thisScore))
    std_List.append(np.std(thisScore))
np.mean(mean_List) #decision tree with max_depth = 4 has a mean accuracy of 0.81324022346368707


################ Gradient Boosting ###############
tuned_parameters = [{'n_estimators': [1500, 2000], 'learning_rate': [0.001, 0.01]}]
model_GB = GridSearchCV(estimator = GradientBoostingClassifier(max_depth=5), 
                   param_grid = tuned_parameters, 
                   cv = 5, scoring='accuracy')
model_GB.fit(xTrain, yTrain)

model_GB.best_estimator_
model_GB.best_score_
model_GB.grid_scores_ 

model_GB = GradientBoostingClassifier(n_estimators = 1500, learning_rate = 0.01, 
                                 max_depth = 5)
variance_bias(xTrain, yTrain, model_GB) 
get_acc(xTrain, yTrain, model_GB)   #0.81526708932270409 
                             
GB_acc = cross_val_score(model_GB, xTrain, yTrain, 
                         cv = 10, scoring = 'accuracy')

np.mean(GB_acc)                        

################ Random Forest ###############
#feat_num = range(3, ncol_train + 1)
#tree_num = [500, 1000, 1500]
depth_num = range(5, 15)
model_RF = GridSearchCV(estimator = RandomForestClassifier(n_estimators = 2000,
                                                           max_features = 'sqrt'), 
                   param_grid = dict(max_depth = depth_num), 
                   cv = 5, scoring='accuracy')
model_RF.fit(xTrain, yTrain)

model_RF.best_estimator_
model_RF.best_score_
model_RF.grid_scores_ # max_depth = 11 with accuracy of 0.82043

model_RF = RandomForestClassifier(n_estimators = 2000, max_features= 'sqrt', max_depth = 11)
variance_bias(xTrain, yTrain, model_RF) 
get_acc(xTrain, yTrain, model_RF) # 0.81728202874898004

model_RF_2 = RandomForestClassifier(n_estimators = 3000, max_features= 6, max_depth = 11)
variance_bias(xTrain, yTrain, model_RF_2)
get_acc(xTrain, yTrain, model_RF) #0.81727073002322526
                                    
RF_acc = cross_val_score(model_GB, xTrain, yTrain, 
                         cv = 10, scoring = 'accuracy')

np.mean(RF_acc) 

y_pred = model_RF.predict(xTest)
y_pred_lt = y_pred.tolist()
out_pred = {'PassengerId':range(892, 1310), 'Survived':y_pred_lt}
out_pred = pd.DataFrame(out_pred)
out_pred.to_csv('test_res.csv', header = True, index = False)

################ lASSO ###############
from sklearn.linear_model import LogisticRegression
alphas = np.logspace(-2, 2, 100)
model_Lasso = GridSearchCV(estimator = LogisticRegression(penalty='l1'), 
                   param_grid = dict(C = alphas), scoring='accuracy', cv = 10)
model_Lasso.fit(xTrain, yTrain)

model_Lasso.best_estimator_.C
model_Lasso.best_score_  
model_Lasso.grid_scores_

model_Lasso = LogisticRegression(penalty='l1', C = model_Lasso.best_estimator_.C)
variance_bias(xTrain, yTrain, model_Lasso) 
get_acc(xTrain, yTrain, model_Lasso)
                                   