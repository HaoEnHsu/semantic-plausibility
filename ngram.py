import pandas as pd
from nltk import ngrams
from collections import Counter

# Load files
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
dev_data = pd.read_csv('dev.csv')

# Separate the labels and the text data into two variables from train data
train_labels = train_data['label']
train_texts = train_data['text']
# Separate the labels and the text data into two variables from test data
test_labels = test_data['label']
test_texts = test_data['text']
# Separate the labels and the text data into two variables from dev data
dev_labels = dev_data['label']
dev_texts = dev_data['text']

# Function to generate n-grams
def generate_ngrams(texts, n):
    ngrams_list = []
    for text in texts:
        tokens = text.split()
        ngrams_list.extend(ngrams(tokens, n))
    return ngrams_list

# Generate bigrams and trigrams
train_bigrams = generate_ngrams(train_texts, 2)
train_trigrams = generate_ngrams(train_texts, 3)

test_bigrams = generate_ngrams(test_texts, 2)
test_trigrams = generate_ngrams(test_texts, 3)

dev_bigrams = generate_ngrams(dev_texts, 2)
dev_trigrams = generate_ngrams(dev_texts, 3)

# Function to count n-grams
def count_ngrams(ngrams_list):
    return Counter(ngrams_list)

# Count bigrams and trigrams
train_bigram_counts = count_ngrams(train_bigrams)
train_trigram_counts = count_ngrams(train_trigrams)

test_bigram_counts = count_ngrams(test_bigrams)
test_trigram_counts = count_ngrams(test_trigrams)

dev_bigram_counts = count_ngrams(dev_bigrams)
dev_trigram_counts = count_ngrams(dev_trigrams)

# Function to sort n-grams by frequency
def sort_ngrams(ngram_counts):
    return sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)

# Sort bigrams and trigrams by frequency
sorted_train_bigrams = sort_ngrams(train_bigram_counts)
sorted_train_trigrams = sort_ngrams(train_trigram_counts)

sorted_test_bigrams = sort_ngrams(test_bigram_counts)
sorted_test_trigrams = sort_ngrams(test_trigram_counts)

sorted_dev_bigrams = sort_ngrams(dev_bigram_counts)
sorted_dev_trigrams = sort_ngrams(dev_trigram_counts)

# Function to list all unique n-grams
def list_unique_ngrams(ngram_counts):
    return list(ngram_counts.keys())

# List all unique bigrams and trigrams
unique_train_bigrams = list_unique_ngrams(train_bigram_counts)
unique_train_trigrams = list_unique_ngrams(train_trigram_counts)

unique_test_bigrams = list_unique_ngrams(test_bigram_counts)
unique_test_trigrams = list_unique_ngrams(test_trigram_counts)

unique_dev_bigrams = list_unique_ngrams(dev_bigram_counts)
unique_dev_trigrams = list_unique_ngrams(dev_trigram_counts)


