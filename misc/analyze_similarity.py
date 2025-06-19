import torch
import numpy as np
import os
import glob
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def load_feature_matrices(feature_dir, layer_name):
    feature_files = sorted(glob.glob(os.path.join(feature_dir, f"features_{layer_name}_batch*.pt")))
    features = []
    for file in feature_files:
        fmap = torch.load(file)  # shape: (B, C, H, W)
        B, C, H, W = fmap.shape
        fmap = fmap.reshape(B, C, -1).mean(dim=2)  # average over spatial dims → (B, C)
        features.append(fmap.numpy())
    return np.vstack(features)  # final shape: (N, C)

def compute_covariance(X):
    X_centered = X - X.mean(axis=0)
    return np.dot(X_centered.T, X_centered) / X.shape[0]

def compute_cosine_similarity(cov1, cov2):
    return cosine_similarity(cov1.flatten().reshape(1, -1), cov2.flatten().reshape(1, -1))[0][0]

def compare_domains(domain1_dir, domain2_dir, layer_name):
    X1 = load_feature_matrices(domain1_dir, layer_name)
    X2 = load_feature_matrices(domain2_dir, layer_name)
    cov1 = compute_covariance(X1)
    cov2 = compute_covariance(X2)
    sim = compute_cosine_similarity(cov1, cov2)
    print(f"[{layer_name}] Cosine similarity: {sim:.4f}")
    return sim

def plot_layerwise_similarity(domain1_dir, domain2_dir, layer_names):
    similarities = []
    for lname in layer_names:
        similarities.append(compare_domains(domain1_dir, domain2_dir, lname))
    
    plt.figure(figsize=(8, 4))
    sns.barplot(x=layer_names, y=similarities)
    plt.ylabel("Cosine Similarity")
    plt.title(f"Feature Similarity between {os.path.basename(domain1_dir)} and {os.path.basename(domain2_dir)}")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_layerwise_similarity("features_finland", "features_poland", ["layer1", "layer2", "layer3", "layer4"])