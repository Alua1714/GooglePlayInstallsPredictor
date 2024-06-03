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
target = 'ModMaximumInstalls'

for column_name in X.columns:
   #if column_name in key_float or (column_name in target_columns and column_name != target):
    #   X = X.drop(columns=[column_name])
    if column_name not in key_floatMod and column_name!=target:
        X = X.drop(columns=[column_name])

X = X.drop(columns=['Released'])
X= X.drop(columns=['ModInstalls'])

y = np.array([])
if target == "ModExit" or target=="Exit":
    y = [1 if element else -1 for element in X[target]]
        
else:
    y = X[target]

X = X.drop(columns=[target])
print(X.columns)

X_train, X_val, y_train, y_val= train_test_split(X, y, test_size=0.2, random_state=42) 
"""
train_df = pd.DataFrame(y_train, columns=['life_expectancy'])
train_df['subset'] = 'train'
train_df = pd.DataFrame({'life_expectancy': y_train, 'subset': 'train'})
val_df = pd.DataFrame({'life_expectancy': y_val, 'subset': 'val'})
life_expectancy_data = pd.concat([train_df, val_df])
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Adjust subplot size for better visibility
sns.histplot(data=life_expectancy_data, x='life_expectancy', hue='subset', kde=True, ax=axs[0], palette='viridis')
axs[0].set_title('Distribution with KDE for All Data')
sns.histplot(data=life_expectancy_data, x='life_expectancy', hue='subset', element='step', ax=axs[1], palette='viridis', fill=True)
axs[1].set_title('Distribution by Subset')
plt.tight_layout()
plt.show()

corr = X_train[X_train.select_dtypes([np.number]).columns].corr()

plt.figure(figsize=(10, 10))
sns.heatmap(corr, center=0, square=True, cbar=True, cmap='coolwarm');
"""
"""
results = pd.DataFrame(columns=['Kernel', 'C', 'epsilon', 'R2', 'MSE', 'median_absolute_error', 'mean_absolute_error'])
knn = KNeighborsRegressor()
knn.fit(X_train,y_train)
y_pred = knn.predict(X_val)
print(comptue_metrics(y_pred,y_val))
results.loc['KNN', :] = ['-', '-', '-'] + comptue_metrics(y_pred,y_val)
sns.scatterplot(x=y_val, y=y_pred)
"""

cv_results = pd.DataFrame(columns=['Kernel', 'C', 'epsilon', 'R2', 'MSE', 'median_absolute_error', 'mean_absolute_error'])

Cs = [10,20,30,40,50,60,100]
epsilons = [0.001,0.0001,0.00001,0.000001,0]
for c in Cs:
    for epsilon in epsilons:
        svm = SVR(kernel='rbf',C=c,epsilon=epsilon)
        svm.fit(X_train,y_train)
        y_pred = svm.predict(X_val)
        y_pred_true = np.array([-1 if a <0 else 1 for a in y_pred])
        cv_results.loc['LinearSVR-{}-{}'.format(c,epsilon), :] = ['RBF', c, epsilon] + comptue_metrics(y_pred,y_val)
        print(cv_results.loc['LinearSVR-{}-{}'.format(c,epsilon), :] )
        
best = cv_results.sort_values(by='R2',ascending=False).iloc[0,:]
print(best)



Cs = [10,20,30,40,50,60,100]
epsilons = [0.001,0.0001,0.00001,0.000001,0]
for c in Cs:
    for epsilon in epsilons:
        svm = SVR(kernel='linear',C=c,epsilon=epsilon)
        svm.fit(X_train,y_train)
        y_pred = svm.predict(X_val)
        y_pred_true = np.array([-1 if a <0 else 1 for a in y_pred])
        cv_results.loc['LinearSVR-{}-{}'.format(c,epsilon), :] = ['Linear', c, epsilon] + comptue_metrics(y_pred,y_val)
        print(cv_results.loc['LinearSVR-{}-{}'.format(c,epsilon), :] )
        
best = cv_results.sort_values(by='R2',ascending=False).iloc[0,:]
print(best)


Cs = [10,20,30,40,50,60,100]
epsilons = [0.001,0.0001,0.00001,0.000001,0]
for c in Cs:
    for epsilon in epsilons:
        svm = SVR(kernel='sigmoid',C=c,epsilon=epsilon)
        svm.fit(X_train,y_train)
        y_pred = svm.predict(X_val)
        y_pred_true = np.array([-1 if a <0 else 1 for a in y_pred])
        cv_results.loc['LinearSVR-{}-{}'.format(c,epsilon), :] = ['Sigmoid', c, epsilon] + comptue_metrics(y_pred,y_val)
        print(cv_results.loc['LinearSVR-{}-{}'.format(c,epsilon), :] )
        
best = cv_results.sort_values(by='R2',ascending=False).iloc[0,:]
print(best)
































