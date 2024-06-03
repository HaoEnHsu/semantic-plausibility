import pandas as pd
import matplotlib.pyplot as plt # allows making plots
from collections import Counter
import nltk # used for wordnet
from nltk.corpus import wordnet as wn


# Hey wait, download WordNet if you don't have it, otherwise comment out line belows, you only have to do once
#nltk.download('wordnet')



def calculate_vocab_statistics(data):
    tokens = [line.split() for line in data['text']] # list of list of tokens
    list_tokens = [word for sublist in tokens for word in sublist] # all tokens in single list
    num_tokens = sum(len(tokens) for tokens in tokens)
    vocab = set(list_tokens)
    token_counts = Counter(list_tokens)
    sorted_word_counts = sorted(token_counts.values(), reverse=True)
    return list_tokens, num_tokens, vocab, sorted_word_counts

def analyze_concreteness(vocabulary): #wordnet is lexical database with words grouped into sets of synonyms that are connected by various relationships
    #such as hypernyms, hyponyms, meronyms, and holonyms, these relationships are indirectly leveraged to infer abstractness
    abstract_count = 0
    concrete_count = 0
    for word in vocabulary:
        synsets = wn.synsets(word) # synsets means sets of cognitive synonyms, retrieves them for given word
        if synsets:
            # Check if any synset of the word is an adjective, if it is, is assumed to be abstract, vast oversimplification, going to work on that
            is_abstract = any('a' in s.pos() for s in synsets)
            if is_abstract:
                abstract_count += 1
            else:
                concrete_count += 1
    total_words = len(vocabulary)
    abstract_percentage = (abstract_count / total_words) * 100
    concrete_percentage = (concrete_count / total_words) * 100
    print("Abstract words: {:.2f}%".format(abstract_percentage))
    print("Concrete words: {:.2f}%".format(concrete_percentage))


#load files
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


# 1. size of train dataset
num_instances = train_data.shape[0]
print("1. size of dataset: \nNumber of data instances in train:", num_instances)
num_instances = test_data.shape[0]
print("Number of data instances in testing data:", num_instances)
num_instances = dev_data.shape[0]
print("Number of data instances in validation data:", num_instances, '\n')

# 2. balance of dataset, how many data instances are labeled plausible versus implausible
label_counts = train_labels.value_counts()
label_counts_dict = label_counts.to_dict()
print("2. Balance of dataset, how many data instances are labeled plausible versus implausible \nTrain label counts:\n", label_counts_dict)

test_label_counts = test_labels.value_counts()
test_label_counts_dict = test_label_counts.to_dict()
print("Test label counts:\n", test_label_counts_dict)

dev_label_counts = dev_labels.value_counts()
dev_label_counts_dict = dev_label_counts.to_dict()
print("Validation label counts:\n", dev_label_counts_dict, '\n')
#making a graph
label_counts_df = pd.DataFrame({
    'Train': label_counts,
    'Test': test_label_counts,
    'Dev': dev_label_counts
}).fillna(0)

plt.figure(figsize=(10, 6))
bar_width = 0.25
index = label_counts_df.index
bar1 = plt.bar(index - bar_width, label_counts_df['Train'], width=bar_width, label='Train', color='blue')
bar2 = plt.bar(index, label_counts_df['Test'], width=bar_width, label='Test', color='orange')
bar3 = plt.bar(index + bar_width, label_counts_df['Dev'], width=bar_width, label='Dev', color='green')
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Counts of Each Label in Train, Test, and Dev Datasets')
plt.xticks(index, index)
plt.show()



# 3. number of words in each data instance
word_counts = train_texts.apply(lambda x: len(x.split()))
word_length_counts = word_counts.value_counts().sort_index()
word_length_dict = word_length_counts.to_dict()
print("3. Number of items(words) in each data instance:\n", word_length_dict,'\n')

plt.figure(figsize=(10, 6)) # makes bar chart
word_length_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Words in Data Instances')
plt.xlabel('Number of Words')
plt.ylabel('Number of Instances')
plt.xticks(rotation=0)
plt.show()


# 4. vocab in dataset, new vocab in test and dev data
train_tokens, total_train_tokens, train_unique_words, train_sorted_word_counts = calculate_vocab_statistics(train_data)
print(f"4. Vocab in dataset and vocab only in dev or test:\nNumber of unique terms in the train dataset: {len(train_unique_words)}")
test_tokens, total_test_tokens, test_unique_words, test_sorted_word_counts = calculate_vocab_statistics(test_data)
print(f"Number of unique terms in the test dataset: {len(test_unique_words)}")
dev_tokens, total_dev_tokens, dev_unique_words, dev_sorted_word_counts = calculate_vocab_statistics(dev_data)
print(f"Number of unique terms in the dev dataset: {len(dev_unique_words)}")
test_new_unique_words = test_unique_words - train_unique_words
dev_new_unique_words = dev_unique_words - train_unique_words
print(f'Number of new unique terms in test set:{len(test_new_unique_words)}')
print(f'Number of new unique terms in dev set:{len(dev_new_unique_words)}\n')


# 5. word count and token frequency
print(f"5. Word count and token frequency:\nNumber of tokens in the train dataset: {total_train_tokens}")
print(f"Number of tokens in the test dataset: {total_test_tokens}")
print(f"Number of tokens in the dev dataset: {total_dev_tokens}", '\n')


# 10. Zipf's law, graph of term frequency
print('10. Zipfs law graph/comparing the data splits(mostly done in 1-5)\n')
# makes plot
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(range(1, len(train_sorted_word_counts) + 1), train_sorted_word_counts)
plt.title('Zipf Law: Train Dataset')
plt.xlabel('Rank')
plt.ylabel('Frequency')

plt.subplot(3, 1, 2)
plt.plot(range(1, len(test_sorted_word_counts) + 1), test_sorted_word_counts)
plt.title('Zipf Law: Test Dataset')
plt.xlabel('Rank')
plt.ylabel('Frequency')

plt.subplot(3, 1, 3)
plt.plot(range(1, len(dev_sorted_word_counts) + 1), dev_sorted_word_counts)
plt.title('Zipf Law: Dev Dataset')
plt.xlabel('Rank')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 11. word concreteness (wordnet, NLTK)
combined_vocab = train_unique_words.union(test_unique_words, dev_unique_words) # unions for sets
combined_tokens = train_tokens + test_tokens + dev_tokens # + for lists
print('11. Word concreteness using wordnet from NLTK:')
print('Terms:')
analyze_concreteness(combined_vocab)
print('Tokens:')
analyze_concreteness(combined_tokens)

