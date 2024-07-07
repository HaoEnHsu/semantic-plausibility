import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, f1_score
import numpy as np
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, texts, labels, features, tokenizer):
        self.texts = texts
        self.labels = labels
        self.features = features
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded_input = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        encoded_input = {k: v.squeeze() for k, v in encoded_input.items()}  # Remove batch dimension
        encoded_input['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        encoded_input['features'] = torch.tensor(self.features[idx], dtype=torch.float)
        return encoded_input

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = torch.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1]
    
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc
    }

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

# Load data
custom_headers = ['label', 'text', 'anim_s', 'anim_o']
train_data = pd.read_csv('train.csv', skiprows=1, header=None, names=custom_headers)
test_data = pd.read_csv('test.csv', skiprows=1, header=None, names=custom_headers)
dev_data = pd.read_csv('dev.csv', skiprows=1, header=None, names=custom_headers)

# Get strings from data to make BERT embeddings
train_strings = get_strings(train_data)
test_strings = get_strings(test_data)
dev_strings = get_strings(dev_data)

# Load labeled data for additional features
train_texts, train_features, train_labels = load_data('train_labeled.csv')
test_texts, test_features, test_labels = load_data('test_labeled.csv')
dev_texts, dev_features, dev_labels = load_data('dev_labeled.csv')

# Weight and sum the animacy features
weighted_train_features = weight_features(train_features, weight=20)
weighted_test_features = weight_features(test_features, weight=20)
weighted_dev_features = weight_features(dev_features, weight=20)

# BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Combine texts, labels, and additional features into datasets
train_dataset = CustomDataset(train_strings, train_labels, weighted_train_features, tokenizer)
test_dataset = CustomDataset(test_strings, test_labels, weighted_test_features, tokenizer)
dev_dataset = CustomDataset(dev_strings, dev_labels, weighted_dev_features, tokenizer)

# Set up Trainer
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Reduce batch size if needed
    per_device_eval_batch_size=8,  # Reduce batch size if needed
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# Check if CUDA is available and use GPU if possible
if torch.cuda.is_available():
    model.to('cuda')

# Train the model
trainer.train()

# Evaluate the model on the test set
test_results = trainer.evaluate(test_dataset)
dev_results = trainer.evaluate(dev_dataset)

print(f"Test Accuracy: {test_results['eval_accuracy']}")
print(f"Dev Accuracy: {dev_results['eval_accuracy']}")
print(f"Test F1 Score: {test_results['eval_f1']}")
print(f"Dev F1 Score: {dev_results['eval_f1']}")
print(f"Test ROC AUC: {test_results['eval_roc_auc']}")
print(f"Dev ROC AUC: {dev_results['eval_roc_auc']}")

# Get predictions and probabilities for ROC curve
test_predictions = trainer.predict(test_dataset)
dev_predictions = trainer.predict(dev_dataset)

test_probs = torch.softmax(torch.tensor(test_predictions.predictions), dim=-1)[:, 1].numpy()
dev_probs = torch.softmax(torch.tensor(dev_predictions.predictions), dim=-1)[:, 1].numpy()

# Compute ROC curve and AUC
fpr_test, tpr_test, _ = roc_curve(test_labels, test_probs)
roc_auc_test = auc(fpr_test, tpr_test)

fpr_dev, tpr_dev, _ = roc_curve(dev_labels, dev_probs)
roc_auc_dev = auc(fpr_dev, tpr_dev)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, color='blue', lw=2, label=f'Test ROC curve (AUC = {roc_auc_test:.2f})')
plt.plot(fpr_dev, tpr_dev, color='red', lw=2, label=f'Dev ROC curve (AUC = {roc_auc_dev:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('BERT AUC-ROC with Animacy')
plt.legend(loc='lower right')
plt.show()
