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

columsToDrop = ['Maximum Installs', 'Price', 'Size', 'Download', 'Last Updated', 'ModInstalls', 'ModMaximumInstalls', 'Rating', 'Exit']
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

Xtrain = datasetTrain.loc[:, datasetTrain.columns != 'ModExit']
Ytrain = datasetTrain['ModExit']

Xtest = datasetTest.loc[:, datasetTest.columns != 'ModExit']
Ytest = datasetTest['ModExit']

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

from sklearn.ensemble import GradientBoostingClassifier
'''
model_GBC = GradientBoostingClassifier().fit(Xtrain, Ytrain)

pred = model_GBC.predict(Xtrain)

print(classification_report(Ytrain, pred, target_names=['1000000.0', '100000.0', '10000.0', '50000.0', '5000.0', '500000.0', '1000.0', '500.0', '100.0', '5000000.0', '50000000.0', '10000000.0', '100000000.0', '1000000000.0', '500000000.0'],))
print('OOB accuracy=', model_GBC.oob_score_)

Ypred = model_GBC.predict(Xval)
print('Validation Accuracy:{}'.format(model_GBC.score(Xval,Yval)))
results.loc['RF-default',:] = compute_metrics(Yval,Ypred)

print(results)
'''


# If you run the above code, you will notice that there is a computational problem here

Xtrain = Xtrain[:20000] 
Ytrain = Ytrain[:20000]
t_n= Ytrain.unique()

model_GBC = GradientBoostingClassifier().fit(Xtrain, Ytrain)

pred = model_GBC.predict(Xtrain)

#print(classification_report(Ytrain, pred, target_names=t_n,))

Ypred = model_GBC.predict(Xval)
print('Validation Accuracy:{}'.format(model_GBC.score(Xval,Yval)))
results.loc['RF-default',:] = compute_metrics(Yval,Ypred)

print(results)

print("End")