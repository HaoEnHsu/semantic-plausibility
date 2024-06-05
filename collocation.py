import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import brown
import nltk
from data_analysis import generate_and_count_ngrams

# Download necessary NLTK data
nltk.download('brown')
nltk.download('punkt')

# Load dataset (reusing the same datasets loaded in ngram.py)
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
dev_data = pd.read_csv('dev.csv')

# Combine text data from all datasets
all_texts = pd.concat([train_data['text'], test_data['text'], dev_data['text']]).tolist()

# Load generated bigrams from ngram.py results
train_bigrams = generate_and_count_ngrams(train_data['text'], 2)
test_bigrams = generate_and_count_ngrams(test_data['text'], 2)
dev_bigrams = generate_and_count_ngrams(dev_data['text'], 2)

# Merge all bigrams from all datasets
dataset_bigrams = list(train_bigrams.elements()) + list(test_bigrams.elements()) + list(dev_bigrams.elements())
dataset_bigram_freq = Counter(dataset_bigrams)

# Generate bigrams from the Brown corpus
brown_bigrams = list(nltk.bigrams(brown.words()))
brown_bigram_freq = Counter(brown_bigrams)

# Function to get frequency of a bigram in the Brown corpus
def get_brown_bigram_freq(bigram):
    return brown_bigram_freq.get(bigram, 0)

# Create a list of bigrams and their frequencies in the Brown corpus
bigrams_with_brown_freq = [(bigram, dataset_bigram_freq[bigram], get_brown_bigram_freq(bigram)) for bigram in dataset_bigram_freq]

# Sort bigrams based on their frequencies in the Brown corpus
sorted_bigrams = sorted(bigrams_with_brown_freq, key=lambda x: x[2], reverse=True)

# Display the sorted bigrams with their frequencies
print("Bigram\tDataset Frequency\tBrown Corpus Frequency")
for bigram, dataset_freq, brown_freq in sorted_bigrams[:20]:
    print(f"{bigram}\t{dataset_freq}\t{brown_freq}")

# Visualization
def plot_bigrams(bigrams, title):
    bigram_strings = [' '.join(bigram[0]) for bigram in bigrams[:20]]  # Only plot top 20 for readability
    dataset_frequencies = [bigram[1] for bigram in bigrams[:20]]
    brown_frequencies = [bigram[2] for bigram in bigrams[:20]]
    
    x = range(len(bigram_strings))
    
    plt.figure(figsize=(15, 5))
    plt.bar(x, dataset_frequencies, width=0.4, label='Dataset', color='blue', align='center')
    plt.bar(x, brown_frequencies, width=0.4, label='Brown Corpus', color='red', align='edge')
    
    plt.xlabel('Bigrams')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(x, bigram_strings, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_bigrams(sorted_bigrams, 'Top 20 Dataset Bigrams Sorted by Brown Corpus Frequency')
