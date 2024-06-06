# Includes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

pd.set_option('display.precision', 3)

# Extra imports
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import graphviz

from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,GradientBoostingClassifier,StackingClassifier,ExtraTreesClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.naive_bayes import GaussianNB

from time import time
from datetime import timedelta

import seaborn as sns
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

np.random.seed(1000)

def compute_metrics(y_true,y_pred):
    accuracy = accuracy_score(y_true,y_pred)
    f1_score_macro = f1_score(y_true,y_pred,average='macro')
    return [accuracy,f1_score_macro]

results = pd.DataFrame(columns=['Accuracy', 'F1-score (macro avg)'])

def confusion(true, pred):
    """
    Function for pretty printing confusion matrices
    """
    pred = pd.Series(pred)
    true = pd.Series(true)
    
    true.name = 'target'
    pred.name = 'predicted'
    cm = pd.crosstab(true.reset_index(drop=True), pred.reset_index(drop=True))
    cm = cm[cm.index]
    return cm

datasetTrain = pd.read_csv('../Dades/X_train_modified.csv')
datasetTest  = pd.read_csv('../Dades/X_test_modified.csv')

datasetTrain['Installs'] = datasetTrain['Installs'].astype('object')
datasetTest['Installs'] = datasetTest['Installs'].astype('object')

columsToDrop = ['Maximum Installs', 'Price', 'Size', 'Download', 'Last Updated', 'ModInstalls', 'ModMaximumInstalls', 'Rating', 'ModExit']
categoricalColumns = ['Installs', 'Category', 'Free', 'Content Rating', 'Ad Supported', 'In App Purchases', 'Editors Choice']

'''
# Convert to categorical variables
for column in categoricalColumns:
    datasetTrain[column] = datasetTrain[column].astype('category')
    datasetTest[column] = datasetTest[column].astype('category')
'''
# Drop repeated or unwanted data (Installs is the catageorial version of "Maximum installs")
datasetTrain = datasetTrain.drop(columns=columsToDrop)
datasetTest  = datasetTest.drop(columns=columsToDrop)

datasetTrain = datasetTrain.drop(columns=['Exit'])
datasetTest = datasetTest.drop(columns=['Exit'])


Xtrain = datasetTrain.loc[:, datasetTrain.columns != 'Installs']
Ytrain = datasetTrain['Installs']

Xtest = datasetTest.loc[:, datasetTest.columns != 'Installs']
Ytest = datasetTest['Installs']


Ytrain = Ytrain.astype('str')
Ytest = Ytest.astype('str')

for column in Xtrain.columns:
        if Xtrain[column].dtype.kind == 'O':
            Xtrain_one_hot = pd.get_dummies(Xtrain[column], prefix=column)
            Xtrain = Xtrain.merge(Xtrain_one_hot,left_index=True,right_index=True)
            Xtrain = Xtrain.drop(columns=[column])
            
for column in Xtest.columns:
        if Xtest[column].dtype.kind == 'O':
            Xtest_one_hot = pd.get_dummies(Xtest[column], prefix=column)
            Xtest = Xtest.merge(Xtest_one_hot,left_index=True,right_index=True)
            Xtest = Xtest.drop(columns=[column])

Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.25, stratify=Ytrain, random_state=1)

print("SETUP DONE")

extra_trees = ExtraTreesClassifier()
extra_trees.fit(Xtrain,Ytrain)

Ypred = extra_trees.predict(Xval)

results.loc['extra_trees',:] = compute_metrics(Yval,Ypred)

results.sort_values(by='F1-score (macro avg)', ascending=False)

print(results)
print("End")

