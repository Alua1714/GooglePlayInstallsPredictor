import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Assuming X_train is already defined and loaded with your data

from pandas import read_csv
# for reproducibility
np.random.seed(7)

X_train = read_csv("../Dades/X_train_sampled.csv", header=0, delimiter=',')
key_float = ['Rating','Installs', 'Price', 'Size', 'Released', 'Last Updated', 'Maximum Installs']
key_floatMod =  ['ModRating','ModInstalls', 'ModPrice', 'ModSize', 'Released', 'ModLast Updated', 'ModMaximum Installs']
print(X_train.columns)


for column_name in X_train.columns:
    if column_name not in key_float:
        X_train = X_train.drop(columns=[column_name])



#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X_train)
# Assuming X_train is already defined and loaded with your data
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X_train)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

def biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    fig, ax = plt.subplots()
    ax.scatter(xs * scalex, ys * scaley, c='blue')  # Color can be changed or made dependent on a variable

    for i in range(coeff.shape[0]):
        # Arrows project features (the variable axies are shown as arrows)
        ax.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='red', alpha=0.5)
        if labels is not None:
            ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='green', ha='center', va='center')

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)
    plt.show()



# Labels for each feature
feature_names = X_train.columns  # Ensure your X_train has a .columns attribute or replace accordingly

# Loadings are the coefficients of the linear combination (in PCA context)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

biplot(X_pca, loadings, labels=feature_names)
