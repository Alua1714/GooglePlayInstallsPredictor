import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.manifold import TSNE
import gower
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import dendrogram, linkage

# for reproducibility
np.random.seed(7)

X = read_csv("../Dades/X_train_sampled.csv", header=0, delimiter=',')
key_float = ['Rating','Installs', 'Price', 'Size', 'Released', 'Last Updated', 'Maximum Installs']
key_floatMod =  ['ModRating','ModInstalls', 'ModPrice', 'ModSize', 'Released', 'ModLast Updated', 'ModMaximum Installs']
print(X.columns)


X_train = X
for column_name in X_train.columns:
    if column_name not in key_float:
        X_train = X_train.drop(columns=[column_name])

print(X_train.columns)


def compute_clustering_and_plot_PCA(K, X_train1, n_components, results):
    #scaler = StandardScaler()
    #scaler.fit(X_train1)
    #X_train = scaler.transform(X_train1)
    # Fem el clustering
    kmeans = KMeans(n_clusters=K, max_iter=1000, random_state=42)
    kmeans.fit(X_train)
    labels = kmeans.labels_

    X_train['Cluster'] = labels
    cluster_averages = X_train.groupby('Cluster').mean()
    for cluster in range(K):
           print(f"Cluster {cluster} Average Values:")
           display(cluster_averages.loc[cluster])
           
    # Fem el PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_train)


    # Femt els PLOTS dels PCA
    columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(principal_components, columns=columns)
    pca_df['Cluster'] = labels
    pairs = list(combinations(columns, 2))
    fig, axes = plt.subplots(len(pairs) // 2 + len(pairs) % 2, 2, figsize=(15, 8))
    for (pc1, pc2), ax in zip(pairs, axes.flatten()):
        sns.scatterplot(x=pc1, y=pc2, hue='Cluster', data=pca_df, palette='viridis', ax=ax, s=100, alpha=0.6)
        ax.set_title(f'Scatter Plot of {pc1} vs {pc2}')
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # Mètriques del profe que no se encara què són
    
    CH = calinski_harabasz_score(X_train, labels),
    S = silhouette_score(X_train, labels),
    DB = davies_bouldin_score(X_train, labels)
    results.loc[K] = [CH,S,DB]
    return [CH,S,DB]

""""
############################

Cosa rara que m'ha donat CHAT GPT una altre manera de reducir dimensionalitat

 ##########################

def compute_clustering_and_plot_tsne(K, X_train,results):
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=K, max_iter=1000)
    kmeans.fit(X_train)
    labels = kmeans.labels_

    # Perform t-SNE for dimensionality reduction to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(X_train)

    # Create a DataFrame for the t-SNE components
    tsne_df = pd.DataFrame(data = tsne_results, columns = ['t-SNE 1', 't-SNE 2'])
    tsne_df['Cluster'] = labels

    # Plot results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='t-SNE 1', y='t-SNE 2', hue='Cluster', data=tsne_df, palette='viridis', s=100)
    plt.title(f't-SNE Clustering with K={K}')
    plt.grid(True)
    plt.show()

    # Store metrics
    results = pd.DataFrame(index=[K], columns=['t-SNE CH', 't-SNE Silhouette', 't-SNE DB'])
    results.loc[K, :] = [calinski_harabasz_score(X_train, labels),
                         silhouette_score(X_train, labels),
                         davies_bouldin_score(X_train, labels)]
    return results



"""

def compute_clustering_and_plot_MDS(K, X_train, results):
    
    #Klustering amb KMeans    
    kmeans = KMeans(n_clusters=K, max_iter=1000)
    kmeans.fit(mds_results)
    labels = kmeans.labels_

    X_train['Cluster'] = labels
    
    #MDS and distància gower
    distance_matrix = gower.gower_matrix(X_train)
    print("hola")
    mds = MDS(n_components=4, dissimilarity='precomputed', random_state=42)
    mds_results = mds.fit_transform(distance_matrix)
    print("ohalkdsalkñdj")
    
    
    # PLOT DELS MDS COMPONENTS
    columns = [f'MDS {i+1}' for i in range(4)]
    mds_df = pd.DataFrame(data=mds_results, columns=columns)
    pairs = combinations(columns, 2)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for (mds1, mds2), ax in zip(pairs, axes.flatten()):
        sns.scatterplot(x=mds1, y=mds2, hue='Cluster', data=mds_df, palette='viridis', ax=ax, s=100, alpha=0.6)
        ax.set_title(f'Scatter Plot of {mds1} vs {mds2}')
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # Mètriques del profe que no se encara què són
    results.loc[K]= [
        calinski_harabasz_score(mds_results, labels),
        silhouette_score(mds_results, labels),
        davies_bouldin_score(mds_results, labels)
    ]
    return results


def perform_clustering_and_plot_dendrogram(data):
    # Ensure the data is in a suitable format, possibly standardize or normalize if required
    # data should be a DataFrame where rows are samples and columns are features
    
    # Generate the linkage matrix using Ward's method
    Z = linkage(data, method='ward')
    
    # Plotting the dendrogram
    plt.figure(figsize=(10, 7))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()



index= pd.MultiIndex.from_arrays([['kmeans'], [3]], names=('model', 'K'))
results_df = pd.DataFrame(columns=['CH score', 'Silhouette score', 'DB score'])

K_values = [2,3,4,5,6,7]


#perform_clustering_and_plot_dendrogram(X_train)


for k in K_values:
    X_compl = X.drop(columns=['Download','Maximum Installs','Price','Rating','Last Updated','Installs','Size'])
    results = compute_clustering_and_plot_PCA(k, X_train,3, results_df)

print(results_df)








