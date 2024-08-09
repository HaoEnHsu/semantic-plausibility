import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class FNN(nn.Module):
    """
    Feedforward Neural Network class.
    """

    def __init__(self):
        """
        Initializes neural network layers.
        """
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(768, 120)  # 768 is BERT's embedding size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(120, 1)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def train_model(self, train_loader, device, num_epochs, dev_loader, patience=5):
        """
        Trains the neural network.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            device (torch.device): Device to use for training.
            num_epochs (int): Number of epochs for training.
            dev_loader (DataLoader): DataLoader for the validation data.

        Returns:
            loss_history (list): List of training losses.
            val_loss_history (list): List of validation losses.
        """
        self.to(device)
        criterion = nn.BCELoss()  # BCELoss since sigmoid is included in forward
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.train()

        loss_history = []  # To store the loss at each epoch
        val_loss_history = []
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(train_loader)
            loss_history.append(epoch_loss)

            val_loss, val_f1, val_accuracy, val_auc, _, _ = self.eval_model(dev_loader, device, dataset_name='Validation')
            val_loss_history.append(val_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(self.state_dict(), 'best_model.pth')  # Save the best model
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs.')
                    break
        # Load the best model
        self.load_state_dict(torch.load('best_model.pth'))
        return loss_history, val_loss_history

    def predict(self, X_test_tensor, device):
        """
        Makes predictions on the test data.

        Args:
            X_test_tensor (torch.Tensor): Test data tensor.
            device (torch.device): Device to use for prediction.

        Returns:
            predictions (torch.Tensor): Predicted labels.
        """
        self.eval()
        with torch.no_grad():
            outputs = self(X_test_tensor.to(device))
            predictions = torch.round(outputs)
        return predictions.cpu()

    def eval_model(self, eval_loader, device, dataset_name):
        """
        Evaluates the model on the given dataset.

        Args:
            eval_loader (DataLoader): DataLoader for the evaluation data.
            device (torch.device): Device to use for evaluation.
            dataset_name (str): Name of the dataset.

        Returns:
            loss (float): Evaluation loss.
            f1 (float): F1 score.
            accuracy (float): Accuracy.
            auc (float): Area under the ROC curve.
            fpr (ndarray): False positive rates.
            tpr (ndarray): True positive rates.
        """
        self.eval()
        criterion = nn.BCELoss()
        loss = 0
        all_labels = []
        all_predictions = []
        all_probs = []
        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
                outputs = self(inputs)
                loss += criterion(outputs, labels).item()
                probs = outputs.cpu().numpy()
                predictions = torch.round(outputs)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probs.extend(probs)
        loss /= len(eval_loader)
        f1 = f1_score(all_labels, all_predictions)
        accuracy = accuracy_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_probs)

        print(f'[{dataset_name}] Loss: {loss}, F1 Score: {f1}, Accuracy: {accuracy}, AUC: {auc}')

        # ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        return loss, f1, accuracy, auc, fpr, tpr

def get_sentence_embeddings(text_list, tokenizer, bert_model, device, batch_size=64):
    """
    Gets BERT sentence embeddings for a list of texts.

    Args:
        text_list (list): List of texts.
        tokenizer (BertTokenizer): BERT tokenizer.
        bert_model (BertModel): BERT model.
        device (torch.device): Device to use for computation.
        batch_size (int): Batch size for processing.

    Returns:
        sentence_embeddings (torch.Tensor): BERT sentence embeddings.
    """
    bert_model.eval()
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
    Extracts text strings from a DataFrame, for BERT embeddings.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.

    Returns:
        list: List of text strings.
    """
    return dataframe['text'].tolist()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load files
custom_headers = ['label', 'text']

# to train with augmented embeddings, follow directions at each 1.

# Load data with custom headers
# 1. Original train embeddings from pep3k data set, to train with augmented dataset, switch train_data
train_data = pd.read_csv('train.csv', skiprows=1, header=None, names=custom_headers)
#train_data = pd.read_csv('data_augmented.csv', skiprows=1, header=None, names=custom_headers)  # Augmented training data
test_data = pd.read_csv('test.csv', skiprows=1, header=None, names=custom_headers)
dev_data = pd.read_csv('dev.csv', skiprows=1, header=None, names=custom_headers)

# Extract labels
y_test = torch.tensor(test_data['label'].values)
y_dev = torch.tensor(dev_data['label'].values)
y_train = torch.tensor(train_data['label'].values)

# 1. Load precomputed BERT embeddings, to use train data, switch train files, last thing
train_sentence_embeddings = torch.load("train_sentence_embeddings.pt")
#train_sentence_embeddings = torch.load("train_augmented_sentence_embeddings.pt")
dev_sentence_embeddings = torch.load("dev_sentence_embeddings.pt")
test_sentence_embeddings = torch.load("test_sentence_embeddings.pt")

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(train_sentence_embeddings, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

dev_dataset = TensorDataset(dev_sentence_embeddings, y_dev)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

test_dataset = TensorDataset(test_sentence_embeddings, y_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize and train the model (if training is needed)
model = FNN()

# Uncomment this section to train the model, and comment out lines 226 and 228
'''
loss_history, val_loss_history = model.train_model(train_loader, device, num_epochs=500, dev_loader=dev_loader)
torch.save(model.state_dict(), 'fnn_baseline_model_aug.pth')

# Plot the loss curve
plt.plot(range(1, len(loss_history) + 1), loss_history, label='Training Loss')
plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.show()
'''

# Load trained model, comment out if training model, choose whether you want model trained on only original data or also augmented data
model.load_state_dict(torch.load('fnn_baseline_model.pth'))
#model.load_state_dict(torch.load('fnn_baseline_model_aug.pth'))
model.to(device)

# Evaluate the model and plot ROC curves
dev_loss, dev_f1, dev_accuracy, dev_auc, dev_fpr, dev_tpr = model.eval_model(dev_loader, device, 'Dev Dataset')
test_loss, test_f1, test_accuracy, test_auc, test_fpr, test_tpr = model.eval_model(test_loader, device, 'Test Dataset')


# Plot ROC curves for Dev and Test datasets in the same figure
plt.figure()
plt.plot(dev_fpr, dev_tpr, color='darkorange', lw=2, label=f'Dev ROC curve (area = {dev_auc:.2f})')
plt.plot(test_fpr, test_tpr, color='fuchsia', lw=2, label=f'Test ROC curve (area = {test_auc:.2f})')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC without Animacy')
plt.legend(loc="lower right")
plt.show()
