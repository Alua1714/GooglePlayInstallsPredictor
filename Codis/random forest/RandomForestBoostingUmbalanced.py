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

datasetTrain = datasetTrain.drop(columns=['Installs'])
datasetTest = datasetTest.drop(columns=['Installs'])

Xtrain = datasetTrain.loc[:, datasetTrain.columns != 'Exit']
Ytrain = datasetTrain['Exit']

Xtest = datasetTest.loc[:, datasetTest.columns != 'Exit']
Ytest = datasetTest['Exit']

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

print("SETUP")
from sklearn.ensemble import AdaBoostClassifier
t_n= Ytrain.unique()

model_GBC = AdaBoostClassifier().fit(Xtrain, Ytrain)

pred = model_GBC.predict(Xtrain)

#print(classification_report(Ytrain, pred, target_names=t_n,))

Ypred = model_GBC.predict(Xval)
print('Validation Accuracy:{}'.format(model_GBC.score(Xval,Yval)))
results.loc['RF-default',:] = compute_metrics(Yval,Ypred)

print(results)

print("End")
