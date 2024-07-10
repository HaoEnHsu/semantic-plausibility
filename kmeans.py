import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve, auc
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt

def load_data(file_path):  
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    tuple: A tuple containing lists of texts, feature arrays, and labels.
    """
    data = pd.read_csv(file_path, header=None)
    texts = data[1].tolist()
    features = data[[2, 3]].values
    labels = data[0].values
    return texts, features, labels

# uncomment line 26 and comment line 29 out if using original data
# train_texts, train_features, train_labels = load_data('train_labeled.csv') 
test_texts, test_features, test_labels = load_data('test_labeled.csv')
dev_texts, dev_features, dev_labels = load_data('dev_labeled.csv')
train_texts, train_features, train_labels = load_data('data_augmented_a.csv') 

def combine_features(features):
    """
    Combine the two features into a single feature by summing them.

    Parameters:
    features (numpy.ndarray): Array of feature values.

    Returns:
    numpy.ndarray: Array of combined feature values.
    """
    return np.sum(features, axis=1, keepdims=True)

train_combined_features = combine_features(train_features)
test_combined_features = combine_features(test_features)
dev_combined_features = combine_features(dev_features)

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(texts):
    """
    Generate BERT embeddings for a list of texts.

    Parameters:
    texts (list): List of text strings.

    Returns:
    numpy.ndarray: Array of BERT embeddings.
    """
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

def perform_kmeans(features, n_clusters=2):
    """
    Apply K-means clustering to the given features.

    Parameters:
    features (numpy.ndarray): Array of feature values.
    n_clusters (int): Number of clusters.

    Returns:
    numpy.ndarray: Array of cluster assignments.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    return clusters

train_clusters = perform_kmeans(train_features_combined)
test_clusters = perform_kmeans(test_features_combined)
dev_clusters = perform_kmeans(dev_features_combined)

def get_labels_from_clusters(clusters, labels):
    """
    Convert cluster assignments to predicted labels based on majority voting.

    Parameters:
    clusters (numpy.ndarray): Array of cluster assignments.
    labels (numpy.ndarray): Array of true labels.

    Returns:
    numpy.ndarray: Array of predicted labels.
    """
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

def evaluate_f1(true_labels, predicted_labels):
    """
    Evaluate F1 score.

    Parameters:
    true_labels (numpy.ndarray): Array of true labels.
    predicted_labels (numpy.ndarray): Array of predicted labels.

    Returns:
    float: F1 score.
    """
    return f1_score(true_labels, predicted_labels)

def evaluate_accuracy(true_labels, predicted_labels):
    """
    Evaluate accuracy.

    Parameters:
    true_labels (numpy.ndarray): Array of true labels.
    predicted_labels (numpy.ndarray): Array of predicted labels.

    Returns:
    float: Accuracy score.
    """
    return accuracy_score(true_labels, predicted_labels)

def evaluate_auc_roc(true_labels, predicted_labels):
    """
    Evaluate AUC-ROC score.

    Parameters:
    true_labels (numpy.ndarray): Array of true labels.
    predicted_labels (numpy.ndarray): Array of predicted labels.

    Returns:
    float: AUC-ROC score.
    """
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

def plot_roc_curve(true_labels, predicted_labels, dataset_name):
    """
    Plot ROC curve.

    Parameters:
    true_labels (numpy.ndarray): Array of true labels.
    predicted_labels (numpy.ndarray): Array of predicted labels.
    dataset_name (str): Name of the dataset being plotted.
    """
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

def perform_kmeans_on_embeddings(embeddings, n_clusters=2):
    """
    Apply K-means clustering to the BERT embeddings.

    Parameters:
    embeddings (numpy.ndarray): Array of BERT embeddings.
    n_clusters (int): Number of clusters.

    Returns:
    numpy.ndarray: Array of cluster assignments.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return clusters

train_clusters_embeddings = perform_kmeans_on_embeddings(train_embeddings)
test_clusters_embeddings = perform_kmeans_on_embeddings(test_embeddings)
dev_clusters_embeddings = perform_kmeans_on_embeddings(dev_embeddings)

train_predicted_labels_embeddings = get_labels_from_clusters(train_clusters_embeddings, train_labels)
test_predicted_labels_embeddings = get_labels_from_clusters(test_clusters_embeddings, test_labels)
dev_predicted_labels_embeddings = get_labels_from_clusters(dev_clusters_embeddings, dev_labels)

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

# Concatenate BERT embeddings with separated animacy features (without combining them)
train_features_separated = np.concatenate((train_embeddings, train_features), axis=1)
test_features_separated = np.concatenate((test_embeddings, test_features), axis=1)
dev_features_separated = np.concatenate((dev_embeddings, dev_features), axis=1)

# Perform K-means clustering
def perform_kmeans(features, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    return clusters

train_clusters_separated = perform_kmeans(train_features_separated)
test_clusters_separated = perform_kmeans(test_features_separated)
dev_clusters_separated = perform_kmeans(dev_features_separated)

# Convert cluster assignments to predicted labels based on majority voting
def get_labels_from_clusters(clusters, labels):
    predicted_labels = np.zeros_like(clusters)
    for i in range(clusters.max() + 1):
        cluster_indices = np.where(clusters == i)[0]
        if len(cluster_indices) > 0:
            majority_label = np.bincount(labels[cluster_indices]).argmax()
            predicted_labels[cluster_indices] = majority_label
    return predicted_labels

train_predicted_labels_separated = get_labels_from_clusters(train_clusters_separated, train_labels)
test_predicted_labels_separated = get_labels_from_clusters(test_clusters_separated, test_labels)
dev_predicted_labels_separated = get_labels_from_clusters(dev_clusters_separated, dev_labels)

# Evaluation functions
def evaluate_f1(true_labels, predicted_labels):
    return f1_score(true_labels, predicted_labels)

def evaluate_accuracy(true_labels, predicted_labels):
    return accuracy_score(true_labels, predicted_labels)

def evaluate_auc_roc(true_labels, predicted_labels):
    return roc_auc_score(true_labels, predicted_labels)

# Evaluate performance
train_f1_separated = evaluate_f1(train_labels, train_predicted_labels_separated)
test_f1_separated = evaluate_f1(test_labels, test_predicted_labels_separated)
dev_f1_separated = evaluate_f1(dev_labels, dev_predicted_labels_separated)

train_accuracy_separated = evaluate_accuracy(train_labels, train_predicted_labels_separated)
test_accuracy_separated = evaluate_accuracy(test_labels, test_predicted_labels_separated)
dev_accuracy_separated = evaluate_accuracy(dev_labels, dev_predicted_labels_separated)

train_auc_roc_separated = evaluate_auc_roc(train_labels, train_predicted_labels_separated)
test_auc_roc_separated = evaluate_auc_roc(test_labels, test_predicted_labels_separated)
dev_auc_roc_separated = evaluate_auc_roc(dev_labels, dev_predicted_labels_separated)

print(f"Train F1-score (Separated Animacy Features): {train_f1_separated:.4f}, Accuracy: {train_accuracy_separated:.4f}, AUC-ROC: {train_auc_roc_separated:.4f}")
print(f"Test F1-score (Separated Animacy Features): {test_f1_separated:.4f}, Accuracy: {test_accuracy_separated:.4f}, AUC-ROC: {test_auc_roc_separated:.4f}")
print(f"Dev F1-score (Separated Animacy Features): {dev_f1_separated:.4f}, Accuracy: {dev_accuracy_separated:.4f}, AUC-ROC: {dev_auc_roc_separated:.4f}")

# Plot ROC curve
def plot_roc_curve(true_labels, predicted_labels, dataset_name):
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels)
    plt.plot(fpr, tpr, label=f'{dataset_name} (AUC = {evaluate_auc_roc(true_labels, predicted_labels):.4f})')

plt.figure(figsize=(10, 6))
plot_roc_curve(train_labels, train_predicted_labels_separated, 'Train (Separated Animacy Features)')
plot_roc_curve(test_labels, test_predicted_labels_separated, 'Test (Separated Animacy Features)')
plot_roc_curve(dev_labels, dev_predicted_labels_separated, 'Dev (Separated Animacy Features)')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K-means AUC-ROC with Separated Animacy Features')
plt.legend()
plt.show()
