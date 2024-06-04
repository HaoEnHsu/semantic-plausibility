import pandas as pd
import nltk
from collections import Counter
import matplotlib.pyplot as plt
import spacy

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

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

# Function to get POS tags using SpaCy
def pos_tag_texts_spacy(texts):
    pos_tags = []
    for text in texts:
        doc = nlp(text)
        pos_tags.extend([(token.text, token.pos_) for token in doc])
    return pos_tags

# Get POS tags for each dataset
train_pos_tags = pos_tag_texts_spacy(train_texts)
test_pos_tags = pos_tag_texts_spacy(test_texts)
dev_pos_tags = pos_tag_texts_spacy(dev_texts)

# Function to calculate POS tag distribution
def pos_tag_distribution(pos_tags):
    tags = [tag for word, tag in pos_tags]
    return Counter(tags)

# Calculate POS tag distribution
train_pos_distribution = pos_tag_distribution(train_pos_tags)
test_pos_distribution = pos_tag_distribution(test_pos_tags)
dev_pos_distribution = pos_tag_distribution(dev_pos_tags)

# Function to plot POS tag distribution
def plot_pos_distribution(pos_distribution, title):
    tags, counts = zip(*pos_distribution.items())
    plt.figure(figsize=(12, 6))
    plt.bar(tags, counts, color='skyblue')
    plt.xlabel('POS Tags')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()

# Plot POS tag distribution for each dataset
plot_pos_distribution(train_pos_distribution, 'Train POS Tag Distribution')
plot_pos_distribution(test_pos_distribution, 'Test POS Tag Distribution')
plot_pos_distribution(dev_pos_distribution, 'Dev POS Tag Distribution')
