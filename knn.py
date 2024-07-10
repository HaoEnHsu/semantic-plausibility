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

    Normalizes word counts in the dictionary values to respective probability distributions. Keys are the same.

    :param word_counts: dict
    :return: dict

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
    strings_for_bert = []
    for index, sentence in enumerate(dataframe['text']):
        strings_for_bert.append (sentence)
    return strings_for_bert


#load files
custom_headers = ['label','text','anim_s','anim_o']
train_data = pd.read_csv('train.csv',skiprows=1, header=None, names=custom_headers)
test_data = pd.read_csv('test.csv', skiprows=1, header=None, names=custom_headers)
dev_data = pd.read_csv('dev.csv', skiprows=1, header=None, names=custom_headers)
train_augmented_data = pd.read_csv('data_augmented.csv',skiprows=1, header=None, names=custom_headers)

# get strings from data to make BERT embeddings
train_strings = get_strings(train_data)
test_strings = get_strings(test_data)
dev_strings = get_strings(dev_data)
train_augmented_strings = get_strings(train_augmented_data)


print(train_strings)
# # BERT
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = BertModel.from_pretrained('bert-base-uncased').to(device)

# # Get train embeddings
# train_augmented_sentence_embeddings = get_sentence_embeddings(train_augmented_strings)
# # train_sentence_embeddings = get_sentence_embeddings(train_strings)
# # print("Train sentence embeddings shape:", train_sentence_embeddings.shape)
# # torch.save(train_augmented_sentence_embeddings, "train_augmented_sentence_embeddings.pt")
# loaded_embeddings = torch.load("train_augmented_sentence_embeddings.pt")
# # print(train_sentence_embeddings)

# # # Get test embeddings
# test_sentence_embeddings = get_sentence_embeddings(test_strings)
# dev_sentence_embeddings = get_sentence_embeddings(dev_strings)
# # X_train = train_sentence_embeddings
# # y_train = list(train_data['label'])

# # X_test = test_sentence_embeddings
# # y_test = list(test_data['label'])

# # rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# # rf_classifier.fit(X_train, y_train)

# # # Predict on the test set
# # y_pred = rf_classifier.predict(X_test)

# # # Evaluate the model
# # accuracy = accuracy_score(y_test, y_pred)
# # print(f"Accuracy: {accuracy}")


# def load_data(file_path):
#     data = pd.read_csv(file_path, header=None)
#     texts = data[1].tolist()
#     features = data[[2, 3]].values
#     labels = data[0].values
#     return texts, features, labels

# train_texts, train_features, train_labels = load_data('train_labeled.csv')
# test_texts, test_features, test_labels = load_data('test_labeled.csv')
# dev_texts, dev_features, dev_labels = load_data('dev_labeled.csv')


# animacy_weight = 20
# new_train_features = []
# for i in train_features:
#     new_train_features.append([sum(i)*animacy_weight])
#     # new_train_features.append([sum(i)])
# # print(new_train_features)

# new_test_features = []
# for i in test_features:
#     new_test_features.append([sum(i)*animacy_weight])
#     # new_test_features.append([sum(i)])
# # print(new_test_features)

# new_dev_features = []
# for i in dev_features:
#     new_dev_features.append([sum(i)*animacy_weight])
#     # new_dev_features.append([sum(i)])



# # Adding animacy features to BERT embeddings
# # train_features_combined = np.concatenate((train_sentence_embeddings, new_train_features),axis=1)
# # test_features_combined = np.concatenate((test_sentence_embeddings, new_test_features), axis=1)
# # dev_features_combined = np.concatenate((dev_sentence_embeddings, new_dev_features), axis=1)
# # print(test_features_combined)


# # KNN Classifier 
# knn1 = KNeighborsClassifier(n_neighbors=1,metric='manhattan')
# knn3 = KNeighborsClassifier(n_neighbors=3,metric='manhattan')
# knn5 = KNeighborsClassifier(n_neighbors=5,metric='manhattan')
# knn7 = KNeighborsClassifier(n_neighbors=7,metric='manhattan')
# knn9 = KNeighborsClassifier(n_neighbors=9,metric='manhattan')
# knn11 = KNeighborsClassifier(n_neighbors=11,metric='manhattan')

# X_train = loaded_embeddings
# y_train = list(train_augmented_data['label'])
# # X_train = train_sentence_embeddings
# # X_train = train_features_combined
# # y_train = list(train_data['label'])

# X_test = test_sentence_embeddings
# # X_test = test_features_combined
# y_test = list(test_data['label'])

# X_dev = dev_sentence_embeddings
# # X_dev = dev_features_combined
# y_dev = list(dev_data['label'])

# # Train KNN

# knn1.fit(X_train, y_train)
# knn3.fit(X_train, y_train)
# knn5.fit(X_train, y_train)
# knn7.fit(X_train, y_train)
# knn9.fit(X_train, y_train)
# knn11.fit(X_train, y_train)


# # Predict 

# y_pred_dev_1 = knn1.predict(X_dev)
# y_pred_dev_3 = knn3.predict(X_dev)
# y_pred_dev_5 = knn5.predict(X_dev)
# y_pred_dev_7 = knn7.predict(X_dev)
# y_pred_dev_9 = knn9.predict(X_dev)
# y_pred_dev_11 = knn11.predict(X_dev)

# y_prob1_dev = knn1.predict_proba(X_dev)[:, 1]
# y_prob3_dev = knn3.predict_proba(X_dev)[:, 1]
# y_prob5_dev = knn5.predict_proba(X_dev)[:, 1]
# y_prob7_dev = knn7.predict_proba(X_dev)[:, 1]
# y_prob9_dev = knn9.predict_proba(X_dev)[:, 1]
# y_prob11_dev = knn11.predict_proba(X_dev)[:, 1]

# y_pred_test_1 = knn1.predict(X_test)
# y_pred_test_3 = knn3.predict(X_test)
# y_pred_test_5 = knn5.predict(X_test)
# y_pred_test_7 = knn7.predict(X_test)
# y_pred_test_9 = knn9.predict(X_test)
# y_pred_test_11 = knn11.predict(X_test)

# y_prob1_test = knn1.predict_proba(X_test)[:, 1]
# y_prob3_test = knn3.predict_proba(X_test)[:, 1]
# y_prob5_test = knn5.predict_proba(X_test)[:, 1]
# y_prob7_test = knn7.predict_proba(X_test)[:, 1]
# y_prob9_test = knn9.predict_proba(X_test)[:, 1]
# y_prob11_test = knn11.predict_proba(X_test)[:, 1]

# # Print accuracy dev
# print("Accuracy DEV with k=1", accuracy_score(y_dev, y_pred_dev_1))
# print("Accuracy DEV with k=3", accuracy_score(y_dev, y_pred_dev_3))
# print("Accuracy DEV with k=5", accuracy_score(y_dev, y_pred_dev_5))
# print("Accuracy DEV with k=7", accuracy_score(y_dev, y_pred_dev_7))
# print("Accuracy DEV with k=9", accuracy_score(y_dev, y_pred_dev_9))
# print("Accuracy DEV with k=11", accuracy_score(y_dev, y_pred_dev_11))
 

# # Print F1 score dev
# print("F1-Score DEV with k=1", f1_score(y_dev, y_pred_dev_1))
# print("F1-Score DEV with k=3", f1_score(y_dev, y_pred_dev_3))
# print("F1-Score DEV with k=5", f1_score(y_dev, y_pred_dev_5))
# print("F1-Score DEV with k=7", f1_score(y_dev, y_pred_dev_7))
# print("F1-Score DEV with k=9", f1_score(y_dev, y_pred_dev_9))
# print("F1-Score DEV with k=11", f1_score(y_dev, y_pred_dev_11))


# # Print accuracy test
# print("Accuracy TEST with k=1", accuracy_score(y_test, y_pred_test_1))
# print("Accuracy TEST with k=3", accuracy_score(y_test, y_pred_test_3))
# print("Accuracy TEST with k=5", accuracy_score(y_test, y_pred_test_5))
# print("Accuracy TEST with k=7", accuracy_score(y_test, y_pred_test_7))
# print("Accuracy TEST with k=9", accuracy_score(y_test, y_pred_test_9))
# print("Accuracy TEST with k=11", accuracy_score(y_test, y_pred_test_11))
 

# # Print F1 score test
# print("F1-Score TEST with k=1", f1_score(y_test, y_pred_test_1))
# print("F1-Score TEST with k=3", f1_score(y_test, y_pred_test_3))
# print("F1-Score TEST with k=5", f1_score(y_test, y_pred_test_5))
# print("F1-Score TEST with k=7", f1_score(y_test, y_pred_test_7))
# print("F1-Score TEST with k=9", f1_score(y_test, y_pred_test_9))
# print("F1-Score TEST with k=11", f1_score(y_test, y_pred_test_11))

# # Print AUC-ROC

# roc_auc_dev = roc_auc_score(y_dev, y_prob9_dev)
# roc_auc_test = roc_auc_score(y_test, y_prob9_test)

# print("AUC-ROC DEV:", roc_auc_dev)
# print("AUC-ROC TEST:", roc_auc_test)


# # # Plot the ROC curve
# # fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_prob9_test)
# # roc_auc_test = auc(fpr_test, tpr_test)

# # fpr_dev, tpr_dev, thresholds_dev = roc_curve(y_dev, y_prob9_dev)
# # roc_auc_dev = auc(fpr_dev, tpr_dev)

# # plt.figure()
# # plt.plot(fpr_dev, tpr_dev, color='darkorange', lw=2, label=f' Dev ROC curve (area = {roc_auc_dev:.2f})')
# # plt.plot(fpr_test, tpr_test, color='fuchsia', lw=2, label=f' Test ROC curve (area = {roc_auc_test:.2f})')
# # plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('KNN AUC-ROC with Animacy')
# # plt.legend(loc="lower right")
# # plt.show()