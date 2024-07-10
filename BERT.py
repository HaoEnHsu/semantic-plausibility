import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim

# Load data
train_df = pd.read_csv('train_labeled.csv')
test_df = pd.read_csv('test_labeled.csv')
dev_df = pd.read_csv('dev_labeled.csv')

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_embeddings(texts):
    """
    Function to get BERT embeddings.

    Parameters:
    texts (list): List of text strings to be embedded.

    Returns:
    numpy.ndarray: Array of BERT embeddings for the input texts.
    """
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()


def preprocess_data(df):
    """
    Function to preprocess data by combining BERT embeddings with additional features.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the dataset.

    Returns:
    torch.Tensor: Concatenated features tensor.
    numpy.ndarray: Labels array.
    """
    texts = df.iloc[:, 1].tolist()
    feature1 = df.iloc[:, 2].values * 20
    feature2 = df.iloc[:, 3].values * 20

    bert_embeddings = get_bert_embeddings(texts)
    concatenated_features = torch.tensor(bert_embeddings)
    concatenated_features = torch.cat((concatenated_features, torch.tensor(feature1).unsqueeze(1), torch.tensor(feature2).unsqueeze(1)), dim=1)

    labels = df.iloc[:, 0].values

    return concatenated_features, labels


train_features, train_labels = preprocess_data(train_df)
test_features, test_labels = preprocess_data(test_df)
dev_features, dev_labels = preprocess_data(dev_df)


class BERTBinaryClassifier(nn.Module):
    """
    BERT-based binary classifier.

    Parameters:
    input_dim (int): Dimension of input features.
    """
    def __init__(self, input_dim):
        super(BERTBinaryClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after sigmoid activation.
        """
        x = self.fc(x)
        return torch.sigmoid(x)


# Define hyperparameters
hyperparameters = {
    'input_dim': train_features.shape[1],
    'learning_rate': 5e-5,
    'num_epochs': 1250,
    'batch_size': 8,
}

# Initialize model, criterion, and optimizer
classifier_model = BERTBinaryClassifier(hyperparameters['input_dim'])
criterion = nn.BCELoss()
optimizer = optim.Adam(classifier_model.parameters(), lr=hyperparameters['learning_rate'])

# Prepare data loaders
train_data = torch.utils.data.TensorDataset(train_features, torch.tensor(train_labels, dtype=torch.float32))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=hyperparameters['batch_size'], shuffle=True)

# Training loop
for epoch in range(hyperparameters['num_epochs']):
    classifier_model.train()
    epoch_loss = 0
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = classifier_model(batch_features).squeeze()
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()


def evaluate_model(model, features, labels):
    """
    Function to evaluate the model.

    Parameters:
    model (nn.Module): Trained model.
    features (torch.Tensor): Feature tensor for evaluation.
    labels (numpy.ndarray): True labels for evaluation.

    Returns:
    tuple: F1 score, ROC-AUC score, and accuracy score.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(features).squeeze()
        preds = (outputs > 0.5).float()
        f1 = f1_score(labels, preds)
        roc_auc = roc_auc_score(labels, outputs)
        accuracy = accuracy_score(labels, preds)
    return f1, roc_auc, accuracy


def plot_combined_roc_curves(model_dev, model_test, features_dev, labels_dev, features_test, labels_test, title):
    """
    Function to plot combined ROC curves for development and test sets.

    Parameters:
    model_dev (nn.Module): Model evaluated on the development set.
    model_test (nn.Module): Model evaluated on the test set.
    features_dev (torch.Tensor): Feature tensor for the development set.
    labels_dev (numpy.ndarray): True labels for the development set.
    features_test (torch.Tensor): Feature tensor for the test set.
    labels_test (numpy.ndarray): True labels for the test set.
    title (str): Title for the ROC plot.
    """
    model_dev.eval()
    model_test.eval()

    with torch.no_grad():
        outputs_dev = model_dev(features_dev).squeeze()
        outputs_test = model_test(features_test).squeeze()

        fpr_dev, tpr_dev, _ = roc_curve(labels_dev, outputs_dev)
        fpr_test, tpr_test, _ = roc_curve(labels_test, outputs_test)

        roc_auc_dev = auc(fpr_dev, tpr_dev)
        roc_auc_test = auc(fpr_test, tpr_test)

        plt.figure()
        plt.plot(fpr_dev, tpr_dev, color='darkorange', lw=2, label=f'Dev ROC curve (area = {roc_auc_dev:.2f})')
        plt.plot(fpr_test, tpr_test, color='blue', lw=2, label=f'Test ROC curve (area = {roc_auc_test:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()


# Evaluate models
f1_dev, roc_auc_dev, accuracy_dev = evaluate_model(classifier_model, dev_features, dev_labels)
f1_test, roc_auc_test, accuracy_test = evaluate_model(classifier_model, test_features, test_labels)

print(f'Dev Set (With Animacy) - F1 Score: {f1_dev}, ROC-AUC: {roc_auc_dev}, Accuracy: {accuracy_dev}')
print(f'Test Set (With Animacy) - F1 Score: {f1_test}, ROC-AUC: {roc_auc_test}, Accuracy: {accuracy_test}')

# Plot combined ROC curve for with animacy features
plot_combined_roc_curves(classifier_model, classifier_model, dev_features, dev_labels, test_features, test_labels, 'BERT AUC-ROC with Animacy')


def preprocess_data_embeddings_only(df):
    """
    Function to preprocess data using BERT embeddings only.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the dataset.

    Returns:
    torch.Tensor: Embeddings tensor.
    numpy.ndarray: Labels array.
    """
    texts = df.iloc[:, 1].tolist()
    bert_embeddings = get_bert_embeddings(texts)
    embeddings_tensor = torch.tensor(bert_embeddings)
    labels = df.iloc[:, 0].values
    return embeddings_tensor, labels


# Preprocess data for embeddings only
train_features_embeddings, train_labels_embeddings = preprocess_data_embeddings_only(train_df)
test_features_embeddings, test_labels_embeddings = preprocess_data_embeddings_only(test_df)
dev_features_embeddings, dev_labels_embeddings = preprocess_data_embeddings_only(dev_df)

# Initialize model and optimizer for embeddings only
input_dim_embeddings = train_features_embeddings.shape[1]
classifier_model_embeddings = BERTBinaryClassifier(input_dim_embeddings)
optimizer_embeddings = optim.Adam(classifier_model_embeddings.parameters(), lr=hyperparameters['learning_rate'])

# Prepare data loader for embeddings only
train_data_embeddings = torch.utils.data.TensorDataset(train_features_embeddings, torch.tensor(train_labels_embeddings, dtype=torch.float32))
train_loader_embeddings = torch.utils.data.DataLoader(train_data_embeddings, batch_size=hyperparameters['batch_size'], shuffle=True)

# Training loop for embeddings only
for epoch in range(hyperparameters['num_epochs']):
    classifier_model_embeddings.train()
    epoch_loss = 0
    for batch_features, batch_labels in train_loader_embeddings:
        optimizer_embeddings.zero_grad()
        outputs = classifier_model_embeddings(batch_features).squeeze()
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer_embeddings.step()
        epoch_loss += loss.item()

# Evaluate models for embeddings only
f1_dev_embeddings, roc_auc_dev_embeddings, accuracy_dev_embeddings = evaluate_model(classifier_model_embeddings, dev_features_embeddings, dev_labels_embeddings)
f1_test_embeddings, roc_auc_test_embeddings, accuracy_test_embeddings = evaluate_model(classifier_model_embeddings, test_features_embeddings, test_labels_embeddings)

print(f'Dev Set (without animacy) - F1 Score: {f1_dev_embeddings}, ROC-AUC: {roc_auc_dev_embeddings}, Accuracy: {accuracy_dev_embeddings}')
print(f'Test Set (without animacy) - F1 Score: {f1_test_embeddings}, ROC-AUC: {roc_auc_test_embeddings}, Accuracy: {accuracy_test_embeddings}')

# Plot combined ROC curve for embeddings only
plot_combined_roc_curves(classifier_model_embeddings, classifier_model_embeddings, dev_features_embeddings, dev_labels_embeddings, test_features_embeddings, test_labels_embeddings, 'BERT AUC-ROC without Animacy')
