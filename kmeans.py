import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertTokenizer, BertModel
import torch

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    texts = data[1].tolist()
    features = data[[2, 3]].values
    labels = data[0].values
    return texts, features, labels

train_texts, train_features, train_labels = load_data('train_labeled.csv')
test_texts, test_features, test_labels = load_data('test_labeled.csv')
dev_texts, dev_features, dev_labels = load_data('dev_labeled.csv')

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states[:, 0, :].numpy()

# Get BERT embeddings
train_embeddings = get_bert_embeddings(train_texts)
test_embeddings = get_bert_embeddings(test_texts)
dev_embeddings = get_bert_embeddings(dev_texts)

# Concatenate BERT embeddings with additional features
train_features_combined = np.concatenate((train_embeddings, train_features), axis=1)
test_features_combined = np.concatenate((test_embeddings, test_features), axis=1)
dev_features_combined = np.concatenate((dev_embeddings, dev_features), axis=1)

# Apply K-means clustering
def perform_kmeans(features, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    return clusters

train_clusters = perform_kmeans(train_features_combined)
test_clusters = perform_kmeans(test_features_combined)
dev_clusters = perform_kmeans(dev_features_combined)

# Convert cluster assignments to predicted labels
def get_labels_from_clusters(clusters, labels):
    predicted_labels = np.zeros_like(clusters)
    for i in range(clusters.max() + 1):
        cluster_indices = np.where(clusters == i)[0]
        if len(cluster_indices) > 0:
            majority_label = np.bincount(labels[cluster_indices]).argmax()
            predicted_labels[cluster_indices] = majority_label
    return predicted_labels

train_predicted_labels = get_labels_from_clusters(train_clusters, train_labels)
test_predicted_labels = get_labels_from_clusters(test_clusters, test_labels)
dev_predicted_labels = get_labels_from_clusters(dev_clusters, dev_labels)

# Evaluate F1 score
def evaluate_f1(true_labels, predicted_labels):
    return f1_score(true_labels, predicted_labels)

# Evaluate accuracy
def evaluate_accuracy(true_labels, predicted_labels):
    return accuracy_score(true_labels, predicted_labels)

train_f1 = evaluate_f1(train_labels, train_predicted_labels)
test_f1 = evaluate_f1(test_labels, test_predicted_labels)
dev_f1 = evaluate_f1(dev_labels, dev_predicted_labels)

train_accuracy = evaluate_accuracy(train_labels, train_predicted_labels)
test_accuracy = evaluate_accuracy(test_labels, test_predicted_labels)
dev_accuracy = evaluate_accuracy(dev_labels, dev_predicted_labels)

print(f"Train F1-score: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}")
print(f"Test F1-score: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}")
print(f"Dev F1-score: {dev_f1:.4f}, Accuracy: {dev_accuracy:.4f}")

# Apply K-means clustering using only BERT embeddings
def perform_kmeans_on_embeddings(embeddings, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return clusters

train_clusters_embeddings = perform_kmeans_on_embeddings(train_embeddings)
test_clusters_embeddings = perform_kmeans_on_embeddings(test_embeddings)
dev_clusters_embeddings = perform_kmeans_on_embeddings(dev_embeddings)

# Convert cluster assignments to predicted labels
train_predicted_labels_embeddings = get_labels_from_clusters(train_clusters_embeddings, train_labels)
test_predicted_labels_embeddings = get_labels_from_clusters(test_clusters_embeddings, test_labels)
dev_predicted_labels_embeddings = get_labels_from_clusters(dev_clusters_embeddings, dev_labels)

# Evaluate F1 score and accuracy
train_f1_embeddings = evaluate_f1(train_labels, train_predicted_labels_embeddings)
test_f1_embeddings = evaluate_f1(test_labels, test_predicted_labels_embeddings)
dev_f1_embeddings = evaluate_f1(dev_labels, dev_predicted_labels_embeddings)

train_accuracy_embeddings = evaluate_accuracy(train_labels, train_predicted_labels_embeddings)
test_accuracy_embeddings = evaluate_accuracy(test_labels, test_predicted_labels_embeddings)
dev_accuracy_embeddings = evaluate_accuracy(dev_labels, dev_predicted_labels_embeddings)

print(f"Train F1-score (Embeddings Only): {train_f1_embeddings:.4f}, Accuracy: {train_accuracy_embeddings:.4f}")
print(f"Test F1-score (Embeddings Only): {test_f1_embeddings:.4f}, Accuracy: {test_accuracy_embeddings:.4f}")
print(f"Dev F1-score (Embeddings Only): {dev_f1_embeddings:.4f}, Accuracy: {dev_accuracy_embeddings:.4f}")


"""from sklearn.cluster import AgglomerativeClustering

# Function to perform Agglomerative Clustering
def perform_agglomerative_clustering(features, n_clusters=2):
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = agg_clustering.fit_predict(features)
    return clusters

# Apply PCA for dimensionality reduction (if not already applied)
from sklearn.decomposition import PCA

pca = PCA(n_components=50, random_state=42)  # Adjust number of components as needed
train_features_pca = pca.fit_transform(train_features_combined)
test_features_pca = pca.transform(test_features_combined)
dev_features_pca = pca.transform(dev_features_combined)

# Apply Agglomerative Clustering
train_clusters_agg = perform_agglomerative_clustering(train_features_pca)
test_clusters_agg = perform_agglomerative_clustering(test_features_pca)
dev_clusters_agg = perform_agglomerative_clustering(dev_features_pca)

# Adjusted label assignment function for Agglomerative Clustering
def get_labels_from_clusters(clusters, labels):
    unique_clusters = np.unique(clusters)
    predicted_labels = np.zeros_like(clusters)
    for cluster in unique_clusters:
        cluster_indices = np.where(clusters == cluster)[0]
        majority_label = np.bincount(labels[cluster_indices]).argmax()
        predicted_labels[cluster_indices] = majority_label
    return predicted_labels

# Get predicted labels
train_predicted_labels_agg = get_labels_from_clusters(train_clusters_agg, train_labels)
test_predicted_labels_agg = get_labels_from_clusters(test_clusters_agg, test_labels)
dev_predicted_labels_agg = get_labels_from_clusters(dev_clusters_agg, dev_labels)

# Evaluate F1 score for Agglomerative Clustering
train_f1_agg = evaluate_f1(train_labels, train_predicted_labels_agg)
test_f1_agg = evaluate_f1(test_labels, test_predicted_labels_agg)
dev_f1_agg = evaluate_f1(dev_labels, dev_predicted_labels_agg)

print("After applying PCA and Agglomerative Clustering:")
print(f"Train F1-score: {train_f1_agg:.4f}")
print(f"Test F1-score: {test_f1_agg:.4f}")
print(f"Dev F1-score: {dev_f1_agg:.4f}")

from sklearn.cluster import SpectralClustering

# Function to perform Spectral Clustering
def perform_spectral_clustering(features, n_clusters=2):
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
    clusters = spectral_clustering.fit_predict(features)
    return clusters

# Apply PCA for dimensionality reduction (if not already applied)
from sklearn.decomposition import PCA

pca = PCA(n_components=50, random_state=42)  # Adjust number of components as needed
train_features_pca = pca.fit_transform(train_features_combined)
test_features_pca = pca.transform(test_features_combined)
dev_features_pca = pca.transform(dev_features_combined)

# Apply Spectral Clustering
train_clusters_spectral = perform_spectral_clustering(train_features_pca)
test_clusters_spectral = perform_spectral_clustering(test_features_pca)
dev_clusters_spectral = perform_spectral_clustering(dev_features_pca)

# Adjusted label assignment function for Spectral Clustering
def get_labels_from_clusters(clusters, labels):
    unique_clusters = np.unique(clusters)
    predicted_labels = np.zeros_like(clusters)
    for cluster in unique_clusters:
        cluster_indices = np.where(clusters == cluster)[0]
        majority_label = np.bincount(labels[cluster_indices]).argmax()
        predicted_labels[cluster_indices] = majority_label
    return predicted_labels

# Get predicted labels
train_predicted_labels_spectral = get_labels_from_clusters(train_clusters_spectral, train_labels)
test_predicted_labels_spectral = get_labels_from_clusters(test_clusters_spectral, test_labels)
dev_predicted_labels_spectral = get_labels_from_clusters(dev_clusters_spectral, dev_labels)

# Evaluate F1 score for Spectral Clustering
train_f1_spectral = evaluate_f1(train_labels, train_predicted_labels_spectral)
test_f1_spectral = evaluate_f1(test_labels, test_predicted_labels_spectral)
dev_f1_spectral = evaluate_f1(dev_labels, dev_predicted_labels_spectral)

print("After applying PCA and Spectral Clustering:")
print(f"Train F1-score: {train_f1_spectral:.4f}")
print(f"Test F1-score: {test_f1_spectral:.4f}")
print(f"Dev F1-score: {dev_f1_spectral:.4f}")"""
