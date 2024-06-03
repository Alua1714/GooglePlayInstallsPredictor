import warnings
warnings.filterwarnings('ignore')
from time import time
from datetime import timedelta
import pandas as pd
import seaborn as sns
import numpy as np
#from dython.nominal import associations
#from dython.nominal import correlation_ratio
#from dython.nominal import cramers_v
from scipy.stats import chi2_contingency 
from scipy.stats import pearsonr 
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error, mean_absolute_error
from sklearn.preprocessing import minmax_scale
from sklearn.svm import LinearSVR, SVR, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt 
from pandas import read_csv
from sklearn.metrics import classification_report


def process_dataframe(X_train, categorical_columns):
    # First, we isolate the numeric columns by removing the categorical ones
    floatColumns = X_train.select_dtypes(include=['float']).columns  
    X_flotcol = X_train[floatColumns]
    # Create dummy variables for the categorical columns only
    X_train_dummies = pd.get_dummies(X_train[categorical_columns], drop_first=True)
    
    # Concatenate the numeric columns and the new dummy variables dataframe
    return pd.concat([X_flotcol, X_train_dummies], axis=1)




def comptue_metrics(y_pred, y_real):
    r2 = r2_score(y_pred,y_real)
    mse = mean_squared_error(y_pred, y_real)
    median_abs_e = median_absolute_error(y_pred, y_real)
    mean_abs_e = mean_absolute_error(y_pred, y_real)
    return [r2, mse, median_abs_e, mean_abs_e]


np.random.seed(7)

X = read_csv("../Dades/X_train_sampled.csv", header=0, delimiter=',')
key_float = ['Rating','Installs', 'Price', 'Size', 'Released', 'Last Updated', 'Maximum Installs']
key_floatMod =  ['ModRating','ModInstalls', 'ModPrice', 'ModSize', 'Released', 'ModLast Updated', 'ModMaximumInstalls']
target_columns = ['Download','ModExit', 'Exit', 'Installs', 'ModInstalls', 'ModMaximumInstalls', 'Maximum Installs']
target = 'ModExit'


for column_name in X.columns:
   if column_name in key_float or (column_name in target_columns and column_name != target):
       X = X.drop(columns=[column_name])
       
       

y = X[target]

X = X.drop(columns=[target])
categorical_columns_all = np.array(['Category', 'Free', 'Content Rating', 'Ad Supported', 'In App Purchases', 'Editors Choice'])#funciona acceptable
categorical_columns_2 = np.array(['Category', 'Ad Supported', 'In App Purchases', 'Editors Choice'])#funciona igual que all
categorical_columns_3 = np.array(['Editors Choice', 'Free'])
X = process_dataframe(X, categorical_columns_all)
print(X.columns)

#NOTES LINEAR KERNER WORKS, OK, RBF workds nice

X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=0.5, random_state=42) 
cv_results = pd.DataFrame(columns=['Kernel', 'C', 'epsilon', 'R2', 'MSE', 'median_absolute_error', 'mean_absolute_error'])


Cs = [10,20,30,40,50,60,100]
epsilons = [0.001,0.0001,0.00001,0.000001,0]
for c in Cs:
    for epsilon in epsilons:
        svm = SVR(kernel='rbf',C=c,epsilon=epsilon)
        svm.fit(X_train,y_train)
        y_pred = svm.predict(X_val)>0.5
        y_val = [1 if el else 0 for el in y_val]
        print(classification_report(y_val, y_pred))












































