import pandas as pd
import nltk
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import brown

nltk.download('brown')
nltk.download('punkt')

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
dev_data = pd.read_csv('dev.csv')

# Concatenate all texts into a single list
all_texts = pd.concat([train_data['text'], test_data['text'], dev_data['text']]).tolist()

# Function to generate bigrams
def generate_bigrams(texts):
    bigrams = []
    for text in texts:
        tokens = nltk.word_tokenize(text)
        bigrams.extend(nltk.bigrams(tokens))
    return bigrams

# Generate bigrams for the dataset
dataset_bigrams = generate_bigrams(all_texts)

# Calculate bigram frequencies in the dataset
dataset_bigram_freq = Counter(dataset_bigrams)

# Generate bigrams from the Brown corpus
brown_bigrams = list(nltk.bigrams(brown.words()))

# Calculate bigram frequencies in the Brown corpus
brown_bigram_freq = Counter(brown_bigrams)

# Function to get frequency of a bigram in the Brown corpus
def get_brown_bigram_freq(bigram):
    return brown_bigram_freq.get(bigram, 0)

# Create a list of bigrams and their frequencies in the Brown corpus
bigrams_with_brown_freq = [(bigram, dataset_bigram_freq[bigram], get_brown_bigram_freq(bigram)) for bigram in dataset_bigram_freq]

# Sort bigrams based on their frequencies in the Brown corpus
sorted_bigrams = sorted(bigrams_with_brown_freq, key=lambda x: x[2], reverse=True)

# Display the sorted bigrams
for bigram, dataset_freq, brown_freq in sorted_bigrams:
    print(f"Bigram: {bigram}, Dataset Frequency: {dataset_freq}, Brown Corpus Frequency: {brown_freq}")

# Visualization
def plot_bigrams(bigrams, title):
    bigram_strings = [' '.join(bigram[0]) for bigram in bigrams[:20]]  
    brown_frequencies = [bigram[2] for bigram in bigrams[:20]]
    
    plt.figure(figsize=(12, 4))
    plt.bar(bigram_strings, brown_frequencies, color='red')
    plt.xlabel('Bigrams')
    plt.ylabel('Frequency in Brown Corpus')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()

plot_bigrams(sorted_bigrams, 'Top 20 Dataset Bigrams Sorted by Brown Corpus Frequency')
