import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
def calculate_corr(target, df, numeric=[], categoric=[]):
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

X = pd.read_csv('../Dades/X_train_modified.csv')
X = X[X['Free'] == False]
floatColumns = X.select_dtypes(include=['float']).columns
categories = []
for column in X.columns:
    if column not in floatColumns and (column!="Maximum Installs" or column != "ModMaximumInstalls"):
        categories.append(column)

        
NotMod = ['Price','Rating','Last Updated','Installs','Size']
Mod = ['ModInstalls', 'ModRating', 'ModPrice', 'ModSize', 'ModLast Updated']

categories = np.array(categories)
calculate_corr("ModMaximumInstalls", X, Mod, categories)
calculate_corr("Installs", X, NotMod, categories)
calculate_corr("ModMaximumInstalls", X, NotMod, categories)
calculate_corr("Installs", X, Mod, categories)




