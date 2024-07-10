import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc


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
    
    # Process each sublist separately
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)

        # Extract the sentence embeddings for the batch
        batch_embeddings = outputs.pooler_output
        all_embeddings.append(batch_embeddings.cpu())  # Move to CPU to save GPU memory

    # Concatenate all batch embeddings
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
    strings_for_bert = []
    for index, sentence in enumerate(dataframe['text']):
        strings_for_bert.append (sentence)
    return strings_for_bert


def load_data(file_path):
    """
    Loads data from a CSV file. The function is used to read added animacy features which will be
    consecutively added to the BERT embeddings. 

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


def weighted_features_separated(features):
    """
    Multiples separated animacy features by a constant.

    Parameters:
    features (list): list of animacy features.

    Retures:
    new_features (list): each element of a list is multiplied by a constant.
    """
    animacy_weight = 20
    new_features = []
    for i in features:
        new_features.append(i*animacy_weight)
    return new_features


def weighted_features_combined(features):
    """
    Multiples combined animacy features by a constant.

    Parameters:
    features (list): list of animacy features.

    Retures:
    new_features (list): each element of a list is multiplied by a constant.
    """
    animacy_weight = 20
    new_features = []
    for i in features:
        new_features.append([sum(i)*animacy_weight])
    return new_features


#Load files
custom_headers = ['label','text','anim_s','anim_o']
train_data = pd.read_csv('train.csv',skiprows=1, header=None, names=custom_headers)
test_data = pd.read_csv('test.csv', skiprows=1, header=None, names=custom_headers)
dev_data = pd.read_csv('dev.csv', skiprows=1, header=None, names=custom_headers)
train_augmented_data = pd.read_csv('data_augmented.csv',skiprows=1, header=None, names=custom_headers)

# Get strings from data to make BERT embeddings
train_strings = get_strings(train_data)
test_strings = get_strings(test_data)
dev_strings = get_strings(dev_data)
train_augmented_strings = get_strings(train_augmented_data)

# BERT

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Get train embeddings
# train_augmented_sentence_embeddings = get_sentence_embeddings(train_augmented_strings)
# train_sentence_embeddings = get_sentence_embeddings(train_strings)
# print("Train sentence embeddings shape:", train_sentence_embeddings.shape)
# torch.save(train_augmented_sentence_embeddings, "train_augmented_sentence_embeddings.pt")

"""
The lines commented above were used to get BERT train embeddings and save them. Depending on whether you are running
a model with augmented train set or not, select one of the loaded embeddings below and comment the other one.
"""
# loaded_embeddings = torch.load("train_augmented_sentence_embeddings.pt")
loaded_embeddings = torch.load("train_sentence_embeddings.pt")

# Get test and dev embeddings
test_sentence_embeddings = get_sentence_embeddings(test_strings)
dev_sentence_embeddings = get_sentence_embeddings(dev_strings)

# Retrieve data from labeled sets used for checking whether animacy improves the model's performance or not
train_augmented_texts, train_augmented_features, train_augmented_labels = load_data('data_augmented_a.csv')
train_texts, train_features, train_labels = load_data('train_labeled.csv')
test_texts, test_features, test_labels = load_data('test_labeled.csv')
dev_texts, dev_features, dev_labels = load_data('dev_labeled.csv')

# Weight animacy features combined for each dataset
new_augmented_train_features_combined = weighted_features_combined(train_augmented_features)
new_train_features_combined = weighted_features_combined(train_features)
new_test_features_combined = weighted_features_combined(test_features)
new_dev_features_combined = weighted_features_combined(dev_features)

# Weight animacy features separated for each dataset
new_augmented_train_features_separated = weighted_features_separated(train_augmented_features)
new_train_features_separated = weighted_features_separated(train_features)
new_test_features_separated = weighted_features_separated(test_features)
new_dev_features_separated = weighted_features_separated(dev_features)

"""
Concatenate BERT embeddings with combined animacy features. If you run the model on the augmented train data, make sure
that you use augmented_train_features_combined variable, and train_features_combined is commented, and vice versa, if
you use the original train data.
"""

# augmented_train_features_combined = np.concatenate((loaded_embeddings, new_augmented_train_features_combined),axis=1)
train_features_combined = np.concatenate((loaded_embeddings, new_train_features_combined),axis=1)
test_features_combined = np.concatenate((test_sentence_embeddings, new_test_features_combined), axis=1)
dev_features_combined = np.concatenate((dev_sentence_embeddings, new_dev_features_combined), axis=1)

"""
Concatenate BERT embeddings with separated animacy features. If you run the model on the augmented train data, make sure
that you use augmented_train_features_separated variable, and train_features_separated is commented, and vice versa, if
you use the original train data.
"""

# augmented_train_features_separated = np.concatenate((loaded_embeddings, new_augmented_train_features_separated), axis=1)
train_features_separated = np.concatenate((loaded_embeddings, new_train_features_separated), axis=1)
test_features_separated = np.concatenate((test_sentence_embeddings, new_test_features_separated), axis=1)
dev_features_separated = np.concatenate((dev_sentence_embeddings, new_dev_features_separated), axis=1)

# KNN Classifier 

knn9 = KNeighborsClassifier(n_neighbors=9,metric='manhattan')

"""
Choosing the right X_train. 

loaded_embeddings - not checking for animacy, regardless of whether the train set is augmented or not.
train_features_combined - checking for animacy with combined animacy features, original train set.
train_features_separated - checking for animacy with separated animacy features, original train set.
augmented_train_features_combined - checking for animacy with combined animacy features, augmented train set.
augmented_train_features_separated - checking for animacy with separated animacy features, augmented train set.

Once you have chosen the correct one, comment all other options.
"""

# X_train = loaded_embeddings
# X_train = train_features_combined
X_train = train_features_separated
# X_train = augmented_train_features_separated
# X_train = augmented_train_features_combined

"""Choose y_train in the same way."""

y_train = list(train_data['label'])
# y_train = list(train_augmented_data['label'])

"""Similarly to choosing X_train, choose the appropriate X_test and make sure others are commented out"""
# X_test = test_sentence_embeddings
# X_test = test_features_combined
X_test = test_features_separated
y_test = list(test_data['label'])

"""Similarly to choosing X_test, choose the appropriate X_dev and make sure others are commented out"""
# X_dev = dev_sentence_embeddings
# X_dev = dev_features_combined
X_dev = dev_features_separated
y_dev = list(dev_data['label'])

# Train KNN

knn9.fit(X_train, y_train)

# Predict 

y_pred_dev_9 = knn9.predict(X_dev)
y_pred_test_9 = knn9.predict(X_test)

# Predict using probabilities (used for AUC-ROC evaluation)
y_prob9_dev = knn9.predict_proba(X_dev)[:, 1]
y_prob9_test = knn9.predict_proba(X_test)[:, 1]

# Print accuracy dev

print("Accuracy DEV with k=9", accuracy_score(y_dev, y_pred_dev_9))
 
# Print F1 score dev

print("F1-Score DEV with k=9", f1_score(y_dev, y_pred_dev_9))

# Print accuracy test

print("Accuracy TEST with k=9", accuracy_score(y_test, y_pred_test_9))

# Print F1 score test

print("F1-Score TEST with k=9", f1_score(y_test, y_pred_test_9))

# Print AUC-ROC

roc_auc_dev = roc_auc_score(y_dev, y_prob9_dev)
roc_auc_test = roc_auc_score(y_test, y_prob9_test)

print("AUC-ROC DEV:", roc_auc_dev)
print("AUC-ROC TEST:", roc_auc_test)

"""Uncomment the below to plot the AUC-ROC"""
# # Plot the ROC curve
# fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_prob9_test)
# roc_auc_test = auc(fpr_test, tpr_test)

# fpr_dev, tpr_dev, thresholds_dev = roc_curve(y_dev, y_prob9_dev)
# roc_auc_dev = auc(fpr_dev, tpr_dev)

# plt.figure()
# plt.plot(fpr_dev, tpr_dev, color='darkorange', lw=2, label=f' Dev ROC curve (area = {roc_auc_dev:.2f})')
# plt.plot(fpr_test, tpr_test, color='fuchsia', lw=2, label=f' Test ROC curve (area = {roc_auc_test:.2f})')
# plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('KNN AUC-ROC')
# plt.legend(loc="lower right")
# plt.show()