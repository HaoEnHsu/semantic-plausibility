import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# Load data, uncomment line 10 and comment line 13 out if train on original data
# train_df = pd.read_csv('train_labeled.csv')
test_df = pd.read_csv('test_labeled.csv')
dev_df = pd.read_csv('dev_labeled.csv')
train_df = pd.read_csv('data_augmented_a.csv')

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


def preprocess_data_with_separate_animacy(df):
    """
    Function to preprocess data by combining BERT embeddings with separate animacy features.

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


def preprocess_data_with_combined_animacy(df):
    """
    Function to preprocess data by combining BERT embeddings with combined animacy feature.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the dataset.

    Returns:
    torch.Tensor: Concatenated features tensor.
    numpy.ndarray: Labels array.
    """
    texts = df.iloc[:, 1].tolist()
    animacy_label = (df.iloc[:, 2].values + df.iloc[:, 3].values).reshape(-1, 1)

    bert_embeddings = get_bert_embeddings(texts)
    concatenated_features = torch.tensor(bert_embeddings)
    concatenated_features = torch.cat((concatenated_features, torch.tensor(animacy_label)), dim=1)

    labels = df.iloc[:, 0].values

    return concatenated_features, labels


def preprocess_data_without_animacy(df):
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


train_features_separate, train_labels_separate = preprocess_data_with_separate_animacy(train_df)
test_features_separate, test_labels_separate = preprocess_data_with_separate_animacy(test_df)
dev_features_separate, dev_labels_separate = preprocess_data_with_separate_animacy(dev_df)

train_features_combined, train_labels_combined = preprocess_data_with_combined_animacy(train_df)
test_features_combined, test_labels_combined = preprocess_data_with_combined_animacy(test_df)
dev_features_combined, dev_labels_combined = preprocess_data_with_combined_animacy(dev_df)

train_features_without, train_labels_without = preprocess_data_without_animacy(train_df)
test_features_without, test_labels_without = preprocess_data_without_animacy(test_df)
dev_features_without, dev_labels_without = preprocess_data_without_animacy(dev_df)


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
    'input_dim': train_features_separate.shape[1],
    'learning_rate': 5e-5,
    'num_epochs': 1000,
    'batch_size': 8,
}

# Initialize models, criterion, and optimizers
model_separate = BERTBinaryClassifier(hyperparameters['input_dim'])
model_combined = BERTBinaryClassifier(train_features_combined.shape[1])
model_without = BERTBinaryClassifier(train_features_without.shape[1])

criterion = nn.BCELoss()

optimizer_separate = optim.Adam(model_separate.parameters(), lr=hyperparameters['learning_rate'])
optimizer_combined = optim.Adam(model_combined.parameters(), lr=hyperparameters['learning_rate'])
optimizer_without = optim.Adam(model_without.parameters(), lr=hyperparameters['learning_rate'])

# Prepare data loaders
train_data_separate = torch.utils.data.TensorDataset(train_features_separate, torch.tensor(train_labels_separate, dtype=torch.float32))
train_loader_separate = torch.utils.data.DataLoader(train_data_separate, batch_size=hyperparameters['batch_size'], shuffle=True)

train_data_combined = torch.utils.data.TensorDataset(train_features_combined, torch.tensor(train_labels_combined, dtype=torch.float32))
train_loader_combined = torch.utils.data.DataLoader(train_data_combined, batch_size=hyperparameters['batch_size'], shuffle=True)

train_data_without = torch.utils.data.TensorDataset(train_features_without, torch.tensor(train_labels_without, dtype=torch.float32))
train_loader_without = torch.utils.data.DataLoader(train_data_without, batch_size=hyperparameters['batch_size'], shuffle=True)


def train_model(model, optimizer, train_loader):
    """
    Function to train the model.

    Parameters:
    model (nn.Module): Model to be trained.
    optimizer (torch.optim.Optimizer): Optimizer for training.
    train_loader (torch.utils.data.DataLoader): DataLoader for training data.
    """
    model.train()
    for epoch in range(hyperparameters['num_epochs']):
        epoch_loss = 0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


train_model(model_separate, optimizer_separate, train_loader_separate)
train_model(model_combined, optimizer_combined, train_loader_combined)
train_model(model_without, optimizer_without, train_loader_without)


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


def plot_roc_curves(model_dev, model_test, features_dev, labels_dev, features_test, labels_test, title):
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
f1_dev_separate, roc_auc_dev_separate, accuracy_dev_separate = evaluate_model(model_separate, dev_features_separate, dev_labels_separate)
f1_test_separate, roc_auc_test_separate, accuracy_test_separate = evaluate_model(model_separate, test_features_separate, test_labels_separate)

print(f'Dev Set (with separate animacy) - F1 Score: {f1_dev_separate}, ROC-AUC: {roc_auc_dev_separate}, Accuracy: {accuracy_dev_separate}')
print(f'Test Set (with separate animacy) - F1 Score: {f1_test_separate}, ROC-AUC: {roc_auc_test_separate}, Accuracy: {accuracy_test_separate}')

# Plot ROC curve for separate animacy features
plot_roc_curves(model_separate, model_separate, dev_features_separate, dev_labels_separate, test_features_separate, test_labels_separate, 'BERT AUC-ROC with Separate Animacy')

# Evaluate models for embeddings only
f1_dev_without, roc_auc_dev_without, accuracy_dev_without = evaluate_model(model_without, dev_features_without, dev_labels_without)
f1_test_without, roc_auc_test_without, accuracy_test_without = evaluate_model(model_without, test_features_without, test_labels_without)

print(f'Dev Set (without animacy) - F1 Score: {f1_dev_without}, ROC-AUC: {roc_auc_dev_without}, Accuracy: {accuracy_dev_without}')
print(f'Test Set (without animacy) - F1 Score: {f1_test_without}, ROC-AUC: {roc_auc_test_without}, Accuracy: {accuracy_test_without}')

# Plot ROC curve for embeddings only
plot_roc_curves(model_without, model_without, dev_features_without, dev_labels_without, test_features_without, test_labels_without, 'BERT AUC-ROC without Animacy')

# Evaluate models with combined animacy label
f1_dev_combined, roc_auc_dev_combined, accuracy_dev_combined = evaluate_model(model_combined, dev_features_combined, dev_labels_combined)
f1_test_combined, roc_auc_test_combined, accuracy_test_combined = evaluate_model(model_combined, test_features_combined, test_labels_combined)

print(f'Dev Set (with combined animacy) - F1 Score: {f1_dev_combined}, ROC-AUC: {roc_auc_dev_combined}, Accuracy: {accuracy_dev_combined}')
print(f'Test Set (with combined animacy) - F1 Score: {f1_test_combined}, ROC-AUC: {roc_auc_test_combined}, Accuracy: {accuracy_test_combined}')

# Plot ROC curve for combined animacy features
plot_roc_curves(model_combined, model_combined, dev_features_combined, dev_labels_combined, test_features_combined, test_labels_combined, 'BERT AUC-ROC with Combined Animacy')

# Save the models
torch.save(model_separate.state_dict(), 'model_separate.pth')
torch.save(model_without.state_dict(), 'model_without.pth')
torch.save(model_combined.state_dict(), 'model_combined.pth')
