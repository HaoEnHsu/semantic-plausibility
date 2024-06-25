import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve, auc
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt

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

# Combine the two features into a single feature
def combine_features(features):
    return np.sum(features, axis=1, keepdims=True)

train_combined_features = combine_features(train_features)
test_combined_features = combine_features(test_features)
dev_combined_features = combine_features(dev_features)

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

# Concatenate BERT embeddings with combined features
train_features_combined = np.concatenate((train_embeddings, train_combined_features), axis=1)
test_features_combined = np.concatenate((test_embeddings, test_combined_features), axis=1)
dev_features_combined = np.concatenate((dev_embeddings, dev_combined_features), axis=1)

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

# Evaluate AUC-ROC
def evaluate_auc_roc(true_labels, predicted_labels):
    return roc_auc_score(true_labels, predicted_labels)

train_f1 = evaluate_f1(train_labels, train_predicted_labels)
test_f1 = evaluate_f1(test_labels, test_predicted_labels)
dev_f1 = evaluate_f1(dev_labels, dev_predicted_labels)

train_accuracy = evaluate_accuracy(train_labels, train_predicted_labels)
test_accuracy = evaluate_accuracy(test_labels, test_predicted_labels)
dev_accuracy = evaluate_accuracy(dev_labels, dev_predicted_labels)

train_auc_roc = evaluate_auc_roc(train_labels, train_predicted_labels)
test_auc_roc = evaluate_auc_roc(test_labels, test_predicted_labels)
dev_auc_roc = evaluate_auc_roc(dev_labels, dev_predicted_labels)

print(f"Train F1-score: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}, AUC-ROC: {train_auc_roc:.4f}")
print(f"Test F1-score: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}, AUC-ROC: {test_auc_roc:.4f}")
print(f"Dev F1-score: {dev_f1:.4f}, Accuracy: {dev_accuracy:.4f}, AUC-ROC: {dev_auc_roc:.4f}")

# Plot AUC-ROC curve
def plot_roc_curve(true_labels, predicted_labels, dataset_name):
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    plt.plot(fpr, tpr, label=f'{dataset_name} (AUC = {evaluate_auc_roc(true_labels, predicted_labels):.4f})')

plt.figure(figsize=(10, 6))
plot_roc_curve(train_labels, train_predicted_labels, 'Train')
plot_roc_curve(test_labels, test_predicted_labels, 'Test')
plot_roc_curve(dev_labels, dev_predicted_labels, 'Dev')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K-means AUC-ROC with animacy')
plt.legend()
plt.show()

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

train_auc_roc_embeddings = evaluate_auc_roc(train_labels, train_predicted_labels_embeddings)
test_auc_roc_embeddings = evaluate_auc_roc(test_labels, test_predicted_labels_embeddings)
dev_auc_roc_embeddings = evaluate_auc_roc(dev_labels, dev_predicted_labels_embeddings)

print(f"Train F1-score (Embeddings Only): {train_f1_embeddings:.4f}, Accuracy: {train_accuracy_embeddings:.4f}, AUC-ROC: {train_auc_roc_embeddings:.4f}")
print(f"Test F1-score (Embeddings Only): {test_f1_embeddings:.4f}, Accuracy: {test_accuracy_embeddings:.4f}, AUC-ROC: {test_auc_roc_embeddings:.4f}")
print(f"Dev F1-score (Embeddings Only): {dev_f1_embeddings:.4f}, Accuracy: {dev_accuracy_embeddings:.4f}, AUC-ROC: {dev_auc_roc_embeddings:.4f}")

# Plot AUC-ROC curve (Embeddings Only)
plt.figure(figsize=(10, 6))
plot_roc_curve(train_labels, train_predicted_labels_embeddings, 'Train (Embeddings Only)')
plot_roc_curve(test_labels, test_predicted_labels_embeddings, 'Test (Embeddings Only)')
plot_roc_curve(dev_labels, dev_predicted_labels_embeddings, 'Dev (Embeddings Only)')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K-means AUC-ROC without animacy')
plt.legend()
plt.show()

# print(dev_features_combined)
# print(dev_embeddings)
