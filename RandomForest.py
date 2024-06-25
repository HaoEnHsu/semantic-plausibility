import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def get_sentence_embeddings(text_list, batch_size=32):
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
    return dataframe['text'].tolist()

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    texts = data[1].tolist()
    features = data[[2, 3]].values
    labels = data[0].values
    return texts, features, labels

def weight_features(features, weight=20):
    return [[sum(f) * weight] for f in features]

def evaluate_model(rf_classifier, X_train, y_train, X_test, y_test, X_dev, y_dev):
    rf_classifier.fit(X_train, y_train)
    y_pred_test = rf_classifier.predict(X_test)
    y_pred_dev = rf_classifier.predict(X_dev)
    
    # Accuracy
    accuracy_test = accuracy_score(y_test, y_pred_test)
    accuracy_dev = accuracy_score(y_dev, y_pred_dev)
    
    # F1 score
    f1_test = f1_score(y_test, y_pred_test)
    f1_dev = f1_score(y_dev, y_pred_dev)
    
    # ROC-AUC
    y_prob_test = rf_classifier.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    
    y_prob_dev = rf_classifier.predict_proba(X_dev)[:, 1]
    fpr_dev, tpr_dev, _ = roc_curve(y_dev, y_prob_dev)
    roc_auc_dev = auc(fpr_dev, tpr_dev)
    
    return accuracy_test, accuracy_dev, f1_test, f1_dev, roc_auc_test, roc_auc_dev, fpr_test, tpr_test, fpr_dev, tpr_dev

# Load files
custom_headers = ['label', 'text', 'anim_s', 'anim_o']
train_data = pd.read_csv('train.csv', skiprows=1, header=None, names=custom_headers)
test_data = pd.read_csv('test.csv', skiprows=1, header=None, names=custom_headers)
dev_data = pd.read_csv('dev.csv', skiprows=1, header=None, names=custom_headers)

# Get strings from data to make BERT embeddings
train_strings = get_strings(train_data)
test_strings = get_strings(test_data)
dev_strings = get_strings(dev_data)

# BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# Get sentence embeddings
train_sentence_embeddings = get_sentence_embeddings(train_strings)
test_sentence_embeddings = get_sentence_embeddings(test_strings)
dev_sentence_embeddings = get_sentence_embeddings(dev_strings)

# Convert BERT embeddings to numpy arrays
X_train = train_sentence_embeddings.numpy()
X_test = test_sentence_embeddings.numpy()
X_dev = dev_sentence_embeddings.numpy()

y_train = train_data['label'].values
y_test = test_data['label'].values
y_dev = dev_data['label'].values

# Evaluate without additional features
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
accuracy_test, accuracy_dev, f1_test, f1_dev, roc_auc_test, roc_auc_dev, fpr_test, tpr_test, fpr_dev, tpr_dev = evaluate_model(
    rf_classifier, X_train, y_train, X_test, y_test, X_dev, y_dev
)

print(f"Test Accuracy (without additional features): {accuracy_test}")
print(f"Dev Accuracy (without additional features): {accuracy_dev}")
print(f"Test F1 Score (without additional features): {f1_test}")
print(f"Dev F1 Score (without additional features): {f1_dev}")
print(f"Test ROC AUC: {roc_auc_test}")
print(f"Dev ROC AUC: {roc_auc_dev}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, color='blue', lw=2, label=f'Test ROC curve (AUC = {roc_auc_test:.2f})')
plt.plot(fpr_dev, tpr_dev, color='red', lw=2, label=f'Dev ROC curve (AUC = {roc_auc_dev:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RF AUC-ROC without Animacy')
plt.legend(loc='lower right')
plt.show()

# Load labeled data for additional features
train_texts, train_features, train_labels = load_data('train_labeled.csv')
test_texts, test_features, test_labels = load_data('test_labeled.csv')
dev_texts, dev_features, dev_labels = load_data('dev_labeled.csv')

# Weight and sum the animacy features
weighted_train_features = weight_features(train_features, weight=20)
weighted_test_features = weight_features(test_features, weight=20)
weighted_dev_features = weight_features(dev_features, weight=20)

# Adding weighted animacy features to BERT embeddings
train_features_combined = np.concatenate((train_sentence_embeddings, weighted_train_features), axis=1)
test_features_combined = np.concatenate((test_sentence_embeddings, weighted_test_features), axis=1)
dev_features_combined = np.concatenate((dev_sentence_embeddings, weighted_dev_features), axis=1)

# Evaluate with additional features
rf_classifier_with_features = RandomForestClassifier(n_estimators=100, random_state=42)
accuracy_test_with_features, accuracy_dev_with_features, f1_test_with_features, f1_dev_with_features, roc_auc_test_with_features, roc_auc_dev_with_features, _, _, _, _ = evaluate_model(
    rf_classifier_with_features, train_features_combined, y_train, test_features_combined, y_test, dev_features_combined, y_dev
)

print(f"Test Accuracy (with additional features): {accuracy_test_with_features}")
print(f"Dev Accuracy (with additional features): {accuracy_dev_with_features}")
print(f"Test F1 Score (with additional features): {f1_test_with_features}")
print(f"Dev F1 Score (with additional features): {f1_dev_with_features}")
print(f"Test ROC AUC (with additional features): {roc_auc_test_with_features}")
print(f"Dev ROC AUC (with additional features): {roc_auc_dev_with_features}")

# Plot ROC curve with additional features
plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, color='blue', lw=2, label=f'Test ROC curve (AUC = {roc_auc_test_with_features:.2f})')
plt.plot(fpr_dev, tpr_dev, color='red', lw=2, label=f'Dev ROC curve (AUC = {roc_auc_dev_with_features:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RF AUC-ROC with Animacy')
plt.legend(loc='lower right')
plt.show()
