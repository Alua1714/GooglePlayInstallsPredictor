import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
def calculate_corr(target, df, numeric=[], categoric=[]):
    # Check if the target and features exist in the dataframe
    if target not in df.columns:
        print("Target variable not found in DataFrame.")
        return
    
    # Calculate correlations for numeric features
    if numeric:
        correlations = df[numeric].corrwith(df[target])

        # Plot each numeric feature against the target variable colored by each categorical feature
        for num_feature in numeric:
            for cat_feature in categoric:
                if cat_feature in df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=num_feature, y=target, hue=df[cat_feature], data=df, palette='bright')
                    plt.title(f'Scatter plot of {num_feature} vs. {target} by {cat_feature}\nCorrelation: {correlations[num_feature]:.2f}')
                    plt.legend(title=cat_feature)
                    plt.show()
                else:
                    print(f"Categorical feature '{cat_feature}' not found in DataFrame.")
                          
                          
def plot_no_corr(target, df):    
    features = df.columns.difference([target])    
    # Plot each feature against the target variable
    for feature in features:
        sns.lmplot(x=feature, y=target, data=df, aspect=1.5)
        plt.title(f'Linear relation between {feature}')
        plt.show()


# Load your dataset
X = pd.read_csv('../Dades/X_train_modified.csv')
X = X[X['Free'] == False]
floatColumns = X.select_dtypes(include=['float']).columns
categories = []
for column in X.columns:
    if column not in floatColumns and (column!="Maximum Installs" or column != "ModMaximumInstalls"):
        categories.append(column)

        
NotMod = ['Price','Rating','Last Updated','Installs','Size']
Mod = ['ModInstalls', 'ModRating', 'ModPrice', 'ModSize', 'ModLast Updated']

# List of all columns excluding the target

categories = np.array(categories)
print(categories)
calculate_corr("ModMaximumInstalls", X, Mod, categories)
calculate_corr("ModMaximumInstalls", X, Mod, categories)


