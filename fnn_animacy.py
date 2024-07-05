import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(769, 120)  # 768 is BERT's embedding size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(120, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def train_model(self, train_loader, device, num_epochs, dev_loader):
        self.to(device)
        criterion = nn.BCELoss()  # BCELoss since sigmoid is included in forward
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.train()

        loss_history = []  # To store the loss at each epoch
        val_loss_history = []

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
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

            val_loss, val_f1, val_accuracy, val_auc, _, _ = self.eval_model(dev_loader, device, dataset_name='Validation')
            val_loss_history.append(val_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}, Validation F1 Score: {val_f1}, Validation Accuracy: {val_accuracy}, Validation AUC: {val_auc}")

        return loss_history, val_loss_history

    def predict(self, X_test_tensor, device):
        self.eval()
        with torch.no_grad():
            outputs = self(X_test_tensor.to(device))
            predictions = torch.round(outputs)
        return predictions.cpu()

    def eval_model(self, eval_loader, device, dataset_name):
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

        print(f'{dataset_name}: Loss: {loss}, F1 Score: {f1}, Accuracy: {accuracy}, AUC: {auc}')

        # ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)

        return loss, f1, accuracy, auc, fpr, tpr

def get_sentence_embeddings(text_list, tokenizer, bert_model, device, batch_size=64):
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
    return dataframe['text'].tolist()

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    texts = data[1].tolist()
    features = data[[2]].values
    labels = data[0].values
    return texts, features, labels

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load files
train_data = pd.read_csv('train.csv', header=None)
test_data = pd.read_csv('test.csv', header=None)
dev_data = pd.read_csv('dev.csv', header=None)

train_texts, train_features, y_train = load_data('train_labeled.csv')
test_texts, test_features, y_test = load_data('test_labeled.csv')
dev_texts, dev_features, y_dev = load_data('dev_labeled.csv')

animacy_weight = 20  # Adjust this weight based on your requirement
train_features[:, -1] *= animacy_weight
test_features[:, -1] *= animacy_weight
dev_features[:, -1] *= animacy_weight

'''
# Ensure the labels and texts are paired correctly
assert len(y_train) == len(train_data['text'])
assert len(y_test) == len(test_data['text'])
assert len(y_dev) == len(dev_data['text'])

# Get strings from data to make BERT embeddings
train_strings = get_strings(train_data)
test_strings = get_strings(test_data)
dev_strings = get_strings(dev_data)

# BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Get embeddings
train_sentence_embeddings = get_sentence_embeddings(train_strings, tokenizer, bert_model, device)
dev_sentence_embeddings = get_sentence_embeddings(dev_strings, tokenizer, bert_model, device)
test_sentence_embeddings = get_sentence_embeddings(test_strings, tokenizer, bert_model, device)

# Save and load embeddings (if needed)
torch.save(train_sentence_embeddings, "train_sentence_embeddings.pt")
torch.save(dev_sentence_embeddings, "dev_sentence_embeddings.pt")
torch.save(test_sentence_embeddings, "test_sentence_embeddings.pt")

'''
train_sentence_embeddings = torch.load("train_sentence_embeddings.pt")
dev_sentence_embeddings = torch.load("dev_sentence_embeddings.pt")
test_sentence_embeddings = torch.load("test_sentence_embeddings.pt")

# Concatenate 'an' vectors with corresponding BERT embeddings
train_combined_embeddings = np.concatenate((train_sentence_embeddings.numpy(), train_features), axis=1)
test_combined_embeddings = np.concatenate((test_sentence_embeddings.numpy(), test_features), axis=1)
dev_combined_embeddings = np.concatenate((dev_sentence_embeddings.numpy(), dev_features), axis=1)

# Convert to float32 before converting to tensors
train_combined_embeddings = torch.tensor(train_combined_embeddings, dtype=torch.float32)
test_combined_embeddings = torch.tensor(test_combined_embeddings, dtype=torch.float32)
dev_combined_embeddings = torch.tensor(dev_combined_embeddings, dtype=torch.float32)

# Convert labels to tensors
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_dev_tensor = torch.tensor(y_dev, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(train_combined_embeddings, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

dev_dataset = TensorDataset(dev_combined_embeddings, y_dev_tensor)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

test_dataset = TensorDataset(test_combined_embeddings, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize and train the model
model = FNN()
'''
loss_history, val_loss_history = model.train_model(train_loader, device, num_epochs=1250, dev_loader=dev_loader)
torch.save(model.state_dict(), 'fnn_model4.pth')

# Plot the loss curve
plt.plot(range(1, len(loss_history) + 1), loss_history, label='Training Loss')
plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.show()
'''
# Load model
model.load_state_dict(torch.load('fnn_model3.pth'))
model.to(device)

# Evaluate the model
dev_loss, dev_f1, dev_accuracy, dev_auc, fpr_dev, tpr_dev = model.eval_model(dev_loader, device, 'Dev Dataset')
test_loss, test_f1, test_accuracy, test_auc, fpr_test, tpr_test = model.eval_model(test_loader, device, 'Test Dataset')

# Plot ROC curve
plt.figure()
plt.plot(fpr_dev, tpr_dev, color='darkorange', lw=2, label='Dev ROC curve (area = %0.2f)' % dev_auc)
plt.plot(fpr_test, tpr_test, color='fuchsia', lw=2, label='Test ROC curve (area = %0.2f)' % test_auc)
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC with Animacy')
plt.legend(loc="lower right")
plt.show()