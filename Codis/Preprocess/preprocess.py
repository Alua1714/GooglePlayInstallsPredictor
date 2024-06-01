#%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from collections import Counter
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.precision', 3)

# extra imports
from scipy.special import boxcox1p
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from scipy.stats import boxcox
from statsmodels.genmod.generalized_linear_model import GLM

# Define exchange rates
exchange_rates = {'USD': 1.0, 'VND': 0.000043, 'GBP': 1.39, 'TRY': 0.12, 'EUR': 1.20}

# Function to convert price to USD
def convert_to_usd(row):
    global exchange_rates
    currency = row['Currency']
    price = row['Price']
    exchange_rate = exchange_rates[currency]
    return round(price * exchange_rate, 2)


def save_dataframe_to_csv(X, filename):
    """
    Saves a given pandas DataFrame to a CSV file.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The name of the file to which the DataFrame will be saved.
    """
    try:
        
        X.to_csv(filename, index=False)  # Set index=False to avoid writing row indices to the file.
        print(f"DataFrame is successfully saved to {filename}")
    except Exception as e:
        print(f"Error occurred: {e}")
        
        
        
# Apply the function to create a new column 'Price_USD'

dataOriginal = read_csv("../Dades/Google-Playstore.csv", header=0, delimiter=',')

dataMissingValues = dataOriginal.dropna()
data20reviews = dataMissingValues[dataMissingValues['Rating Count'] >= 20]
columnsToDrop = ['App Name', 'App Id', 'Rating Count', 'Minimum Android',
       'Minimum Installs', 'Developer Id', 'Developer Website','Developer Email', 
        'Privacy Policy', 'Scraped Time']
dataFilterColumns = data20reviews.drop(columns=columnsToDrop)


all_key = ['Category', 'Rating', 'Installs', 'Free', 'Price', 'Currency', 'Maximum Installs', 'Size', 'Released', 'Last Updated', 'Content Rating', 'Ad Supported', 'In App Purchases', 'Editors Choice']
dataNoMissings1 = dataFilterColumns[dataFilterColumns['Currency'] != 'XXX']
dataNoMissings2 = dataNoMissings1[dataNoMissings1['Size'] != 'Varies with device']

dataNoMissings2['Installs'] = dataNoMissings2['Installs'].str.replace('+', '')
dataNoMissings2['Installs'] = dataNoMissings2['Installs'].str.replace(',', '')
dataNoMissings2['Installs'].astype(float)


dataNoMissings2['Price'] = dataNoMissings2.apply(convert_to_usd, axis=1)
dataTransformed1 = dataNoMissings2.drop(columns=['Currency'])

dataTransformed1['Size'] = dataTransformed1['Size'].str.replace('G', '000000000')
dataTransformed1['Size'] = dataTransformed1['Size'].str.replace('M', '000000')
dataTransformed1['Size'] = dataTransformed1['Size'].str.replace('k', '000')
dataTransformed1['Size'] = dataTransformed1['Size'].str.replace('.', '')
dataTransformed1['Size'] = dataTransformed1['Size'].str.replace(',', '')
dataTransformed1['Size'].astype(float)
    

dataTransformed2 = dataTransformed1
dataTransformed2['Released'] = pd.to_datetime(dataTransformed1['Released'], format='%b %d, %Y')
target_date = pd.to_datetime('Jun 15, 2021')
dataTransformed2['Days_Elapsed'] = (target_date - dataTransformed2['Released']).dt.days
dataTransformed2['Released'] = dataTransformed2['Days_Elapsed']
dataTransformed2 = dataTransformed2.drop(columns=['Days_Elapsed'])

dataTransformed3 = dataTransformed2
dataTransformed3['Last Updated'] = pd.to_datetime(dataTransformed2['Last Updated'], format='%b %d, %Y')
target_date = pd.to_datetime('Jun 15, 2021')
dataTransformed3['Days_Elapsed'] = (target_date - dataTransformed3['Last Updated']).dt.days
dataTransformed3['Last Updated'] = dataTransformed3['Days_Elapsed']
dataTransformed3 = dataTransformed3.drop(columns=['Days_Elapsed'])

key_float = ['Rating', 'Installs', 'Price', 'Size', 'Released', 'Last Updated', 'Maximum Installs']
for i in key_float:
    dataTransformed3[i] = dataTransformed3[i].astype(float)


Y = dataTransformed3["Installs"]
X = dataTransformed3.drop(columns=['Installs'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

X_train["Installs"] = y_train
X_test["Installs"] = y_test

X_train['Download'] = X_train['Installs']/X_train['Released']
X_test['Download'] = X_test['Installs']/X_test['Released']

X_train['LogInstalls'] = np.log(1 + X_train['Installs'])
X_test['LogInstalls'] = np.log(1+X_test['Installs'])

X_train['LogMaximumInstalls'] = np.log(1 + X_train['Maximum Installs'])
X_test['LogMaximumInstalls'] = np.log(1 + X_test['Maximum Installs'])

X_train['LinearizedRating'] = X_train['Rating']**3
X_test['LinearizedRating'] = X_test['Rating']**3

best_lambda = -0.8
X_train['LinearizedPrice'] = boxcox1p(1+X_train['Price'],best_lambda)
X_test['LinearizedPrice'] = boxcox1p(1+X_test['Price'],best_lambda)

best_lambda2=0.5
X_train['LogSize'] = boxcox1p(X_train['Size'],best_lambda2)
X_test['LogSize'] = boxcox1p(X_test['Size'],best_lambda2)

best_lambda3 = 0.25
X_train['LogLast Updated'] = boxcox1p(X_train['Last Updated'],best_lambda3)
X_test['LogLast Updated'] = boxcox1p(X_test['Last Updated'],best_lambda3)

# =============================================================================
#  Normalitzo a N(0,1) només les que hem linearitzat
# =============================================================================

floatColumns = ['Price', 'Released', 'Download', 'LogInstalls', 'LogMaximumInstalls',
       'LinearizedRating', 'LinearizedPrice', 'LogSize', 'LogLast Updated']
#floatColumns = NormalizedData.select_dtypes(include=['float']).columns

for column in floatColumns:
    # Calculate mean and variance
    mean = np.mean(X_train[column])
    variance = np.var(X_train[column])
    X_train[column] = (X_train[column] - mean) / np.sqrt(variance)
    X_test[column] = (X_test[column]-mean) / np.sqrt(variance)
    


# Remove these indices from X_train and y_train


# =============================================================================
# Mètodes de tractament d'ouliers:
# =============================================================================
"""
#QUANTILE EXTRACTION

Q1 = df['feature'].quantile(0.25)
Q3 = df['feature'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_filtered = df[(df['feature'] >= lower_bound) & (df['feature'] <= upper_bound)]


#WINDSORING

df['feature_winsorized'] = mstats.winsorize(df['feature'], limits=[0.05, 0.05])  # 5% at both tails


#IMPUTATION

median = df['feature'].median()
mean = df['feature'].mean()
df.loc[((df['feature'] < lower_bound) | (df['feature'] > upper_bound)), 'feature'] = median

"""


X_train = X_train[X_train['Released'] < 1]
X_test = X_test[X_test['Released'] < 1] #Potser no cal eliminar aquests 
#Potser si, depèn de si volem predir per jocs antics.

Q1 = X_train['LogSize'].quantile(0.25)
Q3 = X_train['LogSize'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
X_train = X_train[(X_train['LogSize'] >= lower_bound) & (X_train['LogSize'] <= upper_bound)]
X_test = X_test[(X_test['LogSize'] >= lower_bound) & (X_test['LogSize'] <= upper_bound)]



save_dataframe_to_csv(X_train, "../Dades/X_train_modified.csv")
save_dataframe_to_csv(X_test, "../Dades/X_test_modified.csv")


sample_fraction = 0.01
sample_size = int(len(X_train) * sample_fraction)
dataSample = X_train.sample(n=sample_size, random_state=42)  # using a fixed seed for reproducibility

save_dataframe_to_csv(dataSample, "../Dades/X_train_sampled.csv")
##OUTLIERS




















