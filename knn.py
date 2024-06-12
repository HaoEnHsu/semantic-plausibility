import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

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
custom_headers = ['label','text']
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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Get train embeddings
train_sentence_embeddings = get_sentence_embeddings(train_strings)
print("Train sentence embeddings shape:", train_sentence_embeddings.shape)
torch.save(train_sentence_embeddings, "train_sentence_embeddings.pt")
loaded_embeddings = torch.load("train_sentence_embeddings.pt")
print(train_sentence_embeddings)
