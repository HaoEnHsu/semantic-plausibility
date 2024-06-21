import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def get_sentence_embeddings(text_list, batch_size=32):
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
    strings_for_bert = []
    for index, sentence in enumerate(dataframe['text']):
        emotion_label = dataframe['label'][index]
        strings_for_bert.append (sentence)
    return strings_for_bert


#load files
custom_headers = ['label','text','anim_s','anim_o']
train_data = pd.read_csv('train.csv',skiprows=1, header=None, names=custom_headers)
test_data = pd.read_csv('test.csv', skiprows=1, header=None, names=custom_headers)
dev_data = pd.read_csv('dev.csv', skiprows=1, header=None, names=custom_headers)

# get strings from data to make BERT embeddings
train_strings = get_strings(train_data)
test_strings = get_strings(test_data)
dev_strings = get_strings(dev_data)

# BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Get train embeddings
train_sentence_embeddings = get_sentence_embeddings(train_strings)
# print("Train sentence embeddings shape:", train_sentence_embeddings.shape)
# torch.save(train_sentence_embeddings, "train_sentence_embeddings.pt")
# loaded_embeddings = torch.load("train_sentence_embeddings.pt")
# print(train_sentence_embeddings)

# # Get test embeddings
test_sentence_embeddings = get_sentence_embeddings(test_strings)

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    texts = data[1].tolist()
    features = data[[2, 3]].values
    labels = data[0].values
    return texts, features, labels

train_texts, train_features, train_labels = load_data('train_labeled.csv')
test_texts, test_features, test_labels = load_data('test_labeled.csv')


new_train_features = []
for i in train_features:
    new_train_features.append([sum(i)])
# print(new_train_features)

# very_new_train_features = []
# for i in new_train_features:
#     if i[0] == 0:
#         very_new_train_features.append([i[0]])
#     else:
#         very_new_train_features.append([i[0]/2])
# # print(very_new_train_features)

new_test_features = []
for i in test_features:
    new_test_features.append([sum(i)])
# print(new_test_features)

# very_new_test_features = []
# for i in new_test_features:
#     if i[0] == 0:
#         very_new_test_features.append([i[0]])
#     else:
#         very_new_test_features.append([i[0]/2])
# # print(very_new_test_features)


# Adding animacy features to BERT embeddings
train_features_combined = np.concatenate((train_sentence_embeddings, new_train_features),axis=1)
test_features_combined = np.concatenate((test_sentence_embeddings, new_test_features), axis=1)

# print(test_features_combined)


# KNN Classifier 
knn1 = KNeighborsClassifier(n_neighbors=1,weights='uniform',)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn7 = KNeighborsClassifier(n_neighbors=7)
knn9 = KNeighborsClassifier(n_neighbors=9)

# X_train = train_sentence_embeddings
X_train = train_features_combined
y_train = list(train_data['label'])

# X_test = test_sentence_embeddings
X_test = test_features_combined
y_test = list(test_data['label'])

# Train KNN

knn1.fit(X_train, y_train)
knn5.fit(X_train, y_train)
knn7.fit(X_train, y_train)
knn9.fit(X_train, y_train)


# Predict 
y_pred_1 = knn1.predict(X_test)
y_pred_5 = knn5.predict(X_test)
y_pred_7 = knn7.predict(X_test)
y_pred_9 = knn9.predict(X_test)


# Print accuracy
print("Accuracy with k=1", accuracy_score(y_test, y_pred_1))
print("Accuracy with k=5", accuracy_score(y_test, y_pred_5))
print("Accuracy with k=7", accuracy_score(y_test, y_pred_7))
print("Accuracy with k=9", accuracy_score(y_test, y_pred_9))


# Print F1 score
print("F1-Score with k=1", f1_score(y_test, y_pred_1))
print("F1-Score with k=5", f1_score(y_test, y_pred_5))
print("F1-Score with k=7", f1_score(y_test, y_pred_7))
print("F1-Score with k=9", f1_score(y_test, y_pred_9))