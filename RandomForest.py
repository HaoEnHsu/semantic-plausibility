import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def get_sentence_embeddings(text_list, batch_size=32):
    """
    Generates sentence embeddings for a list of texts using a BERT model.

    Parameters:
    text_list (list): List of strings to generate embeddings for.
    batch_size (int): Number of texts to process in a batch.

    Returns:
    torch.Tensor: Tensor containing sentence embeddings.
    """
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        batch_embeddings = outputs.pooler_output
        all_embeddings.append(batch_embeddings.cpu())
    sentence_embeddings = torch.cat(all_embeddings, dim=0)
    return sentence_embeddings


def get_strings(dataframe):
    """
    Extracts text data from a dataframe.

    Parameters:
    dataframe (pd.DataFrame): DataFrame containing text data.

    Returns:
    list: List of text strings.
    """
    return dataframe['text'].tolist()


def load_data(file_path):
    """
    Loads data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    tuple: A tuple containing lists of texts, features, and labels.
    """
    data = pd.read_csv(file_path, header=None)
    texts = data[1].tolist()
    features = data[[2, 3]].values
    labels = data[0].values
    return texts, features, labels


def weight_features(features, weight=20):
    """
    Weights and sums feature values.

    Parameters:
    features (np.ndarray): Array of feature values.
    weight (int): Weighting factor.

    Returns:
    list: List of weighted feature sums.
    """
    return [[sum(f) * weight] for f in features]


def evaluate_model(rf_classifier, X_train, y_train, X_test, y_test, X_dev, y_dev):
    """
    Trains and evaluates a RandomForestClassifier.

    Parameters:
    rf_classifier (RandomForestClassifier): Random forest classifier instance.
    X_train (np.array): Training data features.
    y_train (np.array): Training data labels.
    X_test (np.array): Test data features.
    y_test (np.array): Test data labels.
    X_dev (np.array): Development data features.
    y_dev (np.array): Development data labels.

    Returns:
    tuple: Evaluation metrics including accuracy, F1 score, ROC-AUC, and ROC curve values.
    """
    rf_classifier.fit(X_train, y_train)
    y_pred_test = rf_classifier.predict(X_test)
    y_pred_dev = rf_classifier.predict(X_dev)
    
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_dev = accuracy_score(y_dev, y_pred_dev)
    
    f1_test = f1_score(y_test, y_pred_test)
    f1_dev = f1_score(y_dev, y_pred_dev)
    
    y_prob_test = rf_classifier.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    
    y_prob_dev = rf_classifier.predict_proba(X_dev)[:, 1]
    fpr_dev, tpr_dev, _ = roc_curve(y_dev, y_prob_dev)
    roc_auc_dev = auc(fpr_dev, tpr_dev)
    
    return accuracy_test, accuracy_dev, f1_test, f1_dev, roc_auc_test, roc_auc_dev, fpr_test, tpr_test, fpr_dev, tpr_dev


# Load labeled data
test_texts, test_features, test_labels = load_data('test_labeled.csv')
dev_texts, dev_features, dev_labels = load_data('dev_labeled.csv')
train_texts, train_features, train_labels = load_data('data_augmented_a.csv')

# Get strings from data to make BERT embeddings
train_strings = train_texts
test_strings = test_texts
dev_strings = dev_texts

# BERT model and tokenizer setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# Get sentence embeddings
train_embeddings = get_sentence_embeddings(train_strings)
test_embeddings = get_sentence_embeddings(test_strings)
dev_embeddings = get_sentence_embeddings(dev_strings)

# Convert BERT embeddings to numpy arrays
X_train_without = train_embeddings.numpy()
X_test_without = test_embeddings.numpy()
X_dev_without = dev_embeddings.numpy()

y_train = train_labels
y_test = test_labels
y_dev = dev_labels

# Evaluate without additional features
rf_classifier_without = RandomForestClassifier(n_estimators=100, random_state=42)
acc_test_without, acc_dev_without, f1_test_without, f1_dev_without, auc_test_without, auc_dev_without, fpr_test_without, tpr_test_without, fpr_dev_without, tpr_dev_without = evaluate_model(
    rf_classifier_without, X_train_without, y_train, X_test_without, y_test, X_dev_without, y_dev
)

print(f"Test Accuracy (without animacy features): {acc_test_without}")
print(f"Dev Accuracy (without animacy features): {acc_dev_without}")
print(f"Test F1 Score (without animacy features): {f1_test_without}")
print(f"Dev F1 Score (without animacy features): {f1_dev_without}")
print(f"Test ROC AUC (without animacy features): {auc_test_without}")
print(f"Dev ROC AUC (without animacy features): {auc_dev_without}")

# Plot ROC curve without animacy features
plt.figure(figsize=(8, 6))
plt.plot(fpr_test_without, tpr_test_without, color='blue', lw=2, label=f'Test ROC curve (AUC = {auc_test_without:.2f})')
plt.plot(fpr_dev_without, tpr_dev_without, color='red', lw=2, label=f'Dev ROC curve (AUC = {auc_dev_without:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RF AUC-ROC without Animacy')
plt.legend(loc='lower right')
plt.show()

# Weight and sum the animacy features for combined animacy
weighted_train_features = weight_features(train_features, weight=20)
weighted_test_features = weight_features(test_features, weight=20)
weighted_dev_features = weight_features(dev_features, weight=20)

# Adding weighted animacy features to BERT embeddings for combined animacy
X_train_combined = np.concatenate((X_train_without, weighted_train_features), axis=1)
X_test_combined = np.concatenate((X_test_without, weighted_test_features), axis=1)
X_dev_combined = np.concatenate((X_dev_without, weighted_dev_features), axis=1)

# Evaluate with combined animacy features
rf_classifier_combined = RandomForestClassifier(n_estimators=100, random_state=42)
acc_test_combined, acc_dev_combined, f1_test_combined, f1_dev_combined, auc_test_combined, auc_dev_combined, _, _, _, _ = evaluate_model(
    rf_classifier_combined, X_train_combined, y_train, X_test_combined, y_test, X_dev_combined, y_dev
)

print(f"Test Accuracy (with combined animacy features): {acc_test_combined}")
print(f"Dev Accuracy (with combined animacy features): {acc_dev_combined}")
print(f"Test F1 Score (with combined animacy features): {f1_test_combined}")
print(f"Dev F1 Score (with combined animacy features): {f1_dev_combined}")
print(f"Test ROC AUC (with combined animacy features): {auc_test_combined}")
print(f"Dev ROC AUC (with combined animacy features): {auc_dev_combined}")

# Plot ROC curve with combined animacy features
plt.figure(figsize=(8, 6))
plt.plot(fpr_test_without, tpr_test_without, color='blue', lw=2, label=f'Test ROC curve (AUC = {auc_test_combined:.2f})')
plt.plot(fpr_dev_without, tpr_dev_without, color='red', lw=2, label=f'Dev ROC curve (AUC = {auc_dev_combined:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RF AUC-ROC with Combined Animacy')
plt.legend(loc='lower right')
plt.show()

# Adding separate animacy features to BERT embeddings
X_train_separate = np.concatenate((X_train_without, train_features), axis=1)
X_test_separate = np.concatenate((X_test_without, test_features), axis=1)
X_dev_separate = np.concatenate((X_dev_without, dev_features), axis=1)

# Evaluate with separate animacy features
rf_classifier_separate = RandomForestClassifier(n_estimators=100, random_state=42)
acc_test_separate, acc_dev_separate, f1_test_separate, f1_dev_separate, auc_test_separate, auc_dev_separate, fpr_test_separate, tpr_test_separate, fpr_dev_separate, tpr_dev_separate = evaluate_model(
    rf_classifier_separate, X_train_separate, y_train, X_test_separate, y_test, X_dev_separate, y_dev
)

print(f"Test Accuracy (with separate animacy features): {acc_test_separate}")
print(f"Dev Accuracy (with separate animacy features): {acc_dev_separate}")
print(f"Test F1 Score (with separate animacy features): {f1_test_separate}")
print(f"Dev F1 Score (with separate animacy features): {f1_dev_separate}")
print(f"Test ROC AUC (with separate animacy features): {auc_test_separate}")
print(f"Dev ROC AUC (with separate animacy features): {auc_dev_separate}")

# Plot ROC curve with separate animacy features
plt.figure(figsize=(8, 6))
plt.plot(fpr_test_separate, tpr_test_separate, color='blue', lw=2, label=f'Test ROC curve (AUC = {auc_test_separate:.2f})')
plt.plot(fpr_dev_separate, tpr_dev_separate, color='red', lw=2, label=f'Dev ROC curve (AUC = {auc_dev_separate:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RF AUC-ROC with Separate Animacy')
plt.legend(loc='lower right')
plt.show()
