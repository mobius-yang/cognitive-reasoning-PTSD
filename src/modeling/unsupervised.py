# src/modeling/unsupervised.py
"""
implementation of unsupervised learning methods for cognitive reasoning data analysis.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

def run_spectral_clustering(
    df_features: pd.DataFrame, 
    feature_cols: list[str], 
    results_dir: Path,
    n_clusters: int = 3
):

    print("[Unsupervised] Running Spectral Clustering Analysis")

    X = df_features[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    spectral = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='nearest_neighbors', # good for manifold data
        random_state=42,
        n_jobs=1
    )
    labels = spectral.fit_predict(X_scaled)
   
    df_features['Cluster_Label'] = labels
 
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=coords[:, 0], 
        y=coords[:, 1], 
        hue=labels, 
        palette='viridis', 
        s=100,
        style=labels
    )
    plt.title(f'Spectral Clustering Results (N={len(df_features)})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(results_dir / "spectral_clustering_pca.png")
    plt.close()
    
    cluster_means = df_features.groupby('Cluster_Label')[feature_cols].mean()
    cluster_means_norm = (cluster_means - cluster_means.mean()) / cluster_means.std()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means_norm.T, cmap='RdBu_r', center=0, annot=True, fmt='.2f')
    plt.title('Feature Profiles of Spectral Clusters (Z-score)')
    plt.xlabel('Cluster Label')
    plt.tight_layout()
    plt.savefig(results_dir / "spectral_clustering_heatmap.png")
    plt.close()
    
    print(f"Clustering analysis completed. Please check the clustering images in {results_dir}.")
    return df_features
