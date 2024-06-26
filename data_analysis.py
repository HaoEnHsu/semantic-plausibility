import pandas as pd
import matplotlib.pyplot as plt # allows making plots
import nltk # used for wordnet
from collections import Counter
from sklearn.metrics import cohen_kappa_score
from nltk.corpus import wordnet as wn
from nltk import ngrams

# Hey wait, download WordNet if you don't have it, otherwise comment out ine belows, you only have to do once
nltk.download('omw-1.4')
nltk.download('wordnet')


def calculate_vocab_statistics(data): # used in 4, 5, 10
    tokens = [line.split() for line in data['text']] # list of list of tokens
    list_tokens = [word for sublist in tokens for word in sublist] # all tokens in single list
    num_tokens = sum(len(tokens) for tokens in tokens)
    vocab = set(list_tokens)
    token_counts = Counter(list_tokens)
    sorted_word_counts = sorted(token_counts.values(), reverse=True) # sorts by how many tokens per term
    return list_tokens, num_tokens, vocab, sorted_word_counts


def is_abstract_word(word): #wordnet is lexical database with words grouped into sets of synonyms that are connected by various relationships
    # such as hypernyms, hyponyms, meronyms, and holonyms, these relationships are indirectly leveraged to infer abstractness
    synsets = wn.synsets(word) # check if word in synsets, a set of cognitive synonyms
    if not synsets:
        return False

    # If any synset is an adjective, mark as abstract (only applies to small percentage of this dataset, ~2%)
    if any('a' in s.pos() for s in synsets):
        return True

    abstract_keywords = ['idea', 'concept', 'state', 'quality'] # list of keywords associated with abstract concepts
    for synset in synsets: # if keyword in def or example of synset, marked as abstract
        definition = synset.definition()
        examples = synset.examples()

        if any(keyword in definition for keyword in abstract_keywords):
            return True

        if any(keyword in ' '.join(examples) for keyword in abstract_keywords):
            return True

    abstract_hypernyms = {'attribute', 'property', 'state', 'quality'} # set of abstract hypernyms
    # hypernym: word more generic or abstract than its subordinates or hyponyms
    for synset in synsets:
        hypernyms = synset.hypernyms() # if hypernym in synset, marked as abstract
        while hypernyms:
            hypernym_names = {hypernym.name().split('.')[0] for hypernym in hypernyms}
            if abstract_hypernyms & hypernym_names:
                return True
            hypernyms = [h for hypernym in hypernyms for h in hypernym.hypernyms()]

    return False

# Total number of all words. Used in 6, 12. 

def all_words(pd_file):
    list_of_all_words = list()
    for i in pd_file['text']:
        i = i.split()
        for j in i:
            list_of_all_words.append(j)
    return list_of_all_words

# Function extracting a set of unique words from a file. Used in 6, 12.

def vocabulary(pd_file):
    unique_words = set()
    for i in pd_file['text']:
        i = i.split()
        for j in i: 
           unique_words.add(j)
    return unique_words

# Function extracting words that are out of vocabulary in train data. Used in 6.
def out_of_vocabulary(vocabulary):
    out_of_vocab = set()
    for word in vocabulary:
        if word not in vocabulary_train:
            out_of_vocab.add(word)
    return out_of_vocab

# Function computing the number of words that are both subject and object. Used in 12.
def subject_and_object(pd_file):
    instances = pd_file['text'] 
    subject = []
    verb = [] 
    object = []
    for instance in instances:
        instance = instance.split()
        subject.append(instance[0])
        verb.append(instance[1])
        object.append(instance[2])

    subject_object = set()
    for i in subject:
        if i in object:
            subject_object.add(i)
    return len(subject_object)


#Function computing the number of words that are both subject and verb. Used in 12.
def subject_and_verb(pd_file):
    instances = pd_file['text'] 
    subject = []
    verb = [] 
    object = []
    for instance in instances:
        instance = instance.split()
        subject.append(instance[0])
        verb.append(instance[1])
        object.append(instance[2])

    subject_verb = set()
    for i in subject:
        if i in verb:
            subject_verb.add(i)
    return len(subject_verb)

#Function computing the number of words that are both verb and object. Used in 12.
def verb_and_object(pd_file):
    instances = pd_file['text'] 
    subject = []
    verb = [] 
    object = []
    for instance in instances:
        instance = instance.split()
        subject.append(instance[0])
        verb.append(instance[1])
        object.append(instance[2])

    verb_object = set()
    for i in verb:
        if i in object:
            verb_object.add(i)
    return len(verb_object)

# Function printing the number of masculine, feminine, and neutral nouns in each set. Used in 13.

def gendered_nouns(pd_file):
    masculine_count = 0
    feminine_count = 0
    gender_neutral_count = 0
    for i in all_words(pd_file):
        if i in masculine_nouns:
            masculine_count += 1
        elif i in feminine_nouns:
            feminine_count += 1
        elif i in gender_neutral_nouns:
            gender_neutral_count += 1
    total_sum = masculine_count + feminine_count + gender_neutral_count
    print("Number of masculine nouns:", masculine_count, ", Percentage among gender-specific nouns:", masculine_count/total_sum)
    print("Number of feminine nouns:", feminine_count, ", Percentage among gender-specific nouns:", feminine_count/total_sum)
    print("Number of neutral nouns:", gender_neutral_count, ", Percentage among gender-specific nouns:", gender_neutral_count/total_sum)

def analyze_concreteness(vocabulary):
    abstract_count = 0
    concrete_count = 0
    not_in_wordnet_count = 0
    abstract_words = []

    for word in vocabulary: # loop putting each word in cat
        synsets = wn.synsets(word)
        if not synsets:
            not_in_wordnet_count += 1
        elif is_abstract_word(word):
            abstract_count += 1
            abstract_words.append(word)
        else:
            concrete_count += 1

    total_words = len(vocabulary)
    abstract_percentage = (abstract_count / total_words) * 100
    concrete_percentage = (concrete_count / total_words) * 100
    not_in_wordnet_percentage = (not_in_wordnet_count / total_words) * 100
    print("Words that can be abstract: {:.2f}%".format(abstract_percentage))
    print("Concrete words: {:.2f}%".format(concrete_percentage))
    print("Words not in WordNet: {:.2f}%".format(not_in_wordnet_percentage))
    # print("\nAbstract words found:")
    # print(", ".join(abstract_words))

# Function to generate and count n-grams, used in 7 and 10
def generate_and_count_ngrams(texts, n):
    ngrams_list = [ngram for text in texts for ngram in ngrams(text.split(), n)]
    return Counter(ngrams_list)

# Function to get unique n-grams, used in 7
def unique_ngrams(ngram_counts):
    return list(ngram_counts.keys())


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

#6. Out-of-vocabulary words in dev and test

# Instantiating vocabularies of train, dev, and test sets.
vocabulary_train = vocabulary(train_data)
vocabulary_dev = vocabulary(dev_data)
vocabulary_test = vocabulary(test_data)

oov_dev = out_of_vocabulary(vocabulary_dev)
oov_test = out_of_vocabulary(vocabulary_test)

print("6. Out-of-vocabulary words")
print("Words in dev set that do not appear in train set: ",oov_dev)
print("Number of OOV words in dev set:",len(oov_dev), "\nDev OOV Percentage:",len(oov_dev)/len(vocabulary_dev)) #Output: 13, 0.039
print("Words in test set that do not appear in train set:", oov_test)
print("Number of OOV words in test set:",len(oov_test), "\nTest OOV Percentage:",len(oov_test)/len(vocabulary_test)) #Output: 12, 0.038

# 7. n-grams; the count of unique bigrams and trigrams

results = {}   # Dictionary to store results
datasets = {'train': train_data['text'], 'test': test_data['text'], 'dev': dev_data['text']}

# Process each dataset
for name, texts in datasets.items():
    results[f'{name}_bigrams'] = generate_and_count_ngrams(texts, 2)
    results[f'{name}_trigrams'] = generate_and_count_ngrams(texts, 3)
    results[f'unique_{name}_bigrams'] = unique_ngrams(results[f'{name}_bigrams'])
    results[f'unique_{name}_trigrams'] = unique_ngrams(results[f'{name}_trigrams'])

# Print unique n-gram counts
for name in datasets.keys():
    print(f"number of unique {name} bigram:", len(results[f'unique_{name}_bigrams']))
    print(f"number of unique {name} trigram:", len(results[f'unique_{name}_trigrams']))

# 8. Zipf's law, graph of term frequency
print('8. Zipfs law graph/comparing the data splits(mostly done in 1-5)\n')
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

# 9. (Dis)agreement in the annotation of train data. Cohen's kappa.

# Reading from the annotation_agreement.csv

annotation_data = pd.read_csv("annotation agreement.csv")

clara_anot = annotation_data['Clara']
sergei_anot = annotation_data['Sergei']
shawn_anot = annotation_data['Shawn']
gold_standard = annotation_data['gold_standard']

clara_gold = (cohen_kappa_score(clara_anot,gold_standard))
sergei_gold = (cohen_kappa_score(sergei_anot,gold_standard))
shawn_gold = (cohen_kappa_score(shawn_anot,gold_standard))

aver_kappa_score = (clara_gold + sergei_gold + shawn_gold)/3
print("9. Cohen's kappa")
print("Clara vs Gold:", clara_gold)
print("Sergei vs Gold:", sergei_gold)
print("Hao-En vs Gold:", shawn_gold)
print("Average kappa score: ", aver_kappa_score)


# 12. Number of unique words that appear in the dataset as both subject and object (S-O), subject and verb (S-V), verb and object (V-O).

print("12. Number of unique words that appear in the dataset as both subject and object (S-O), subject and verb (S-V), verb and object (V-O).")
print("Subject-Object:")
print(subject_and_object(train_data))
print(subject_and_object(dev_data))
print(subject_and_object(test_data))
print("Subject-Verb:")
print(subject_and_verb(train_data))
print(subject_and_verb(dev_data))
print(subject_and_verb(test_data))
print("Verb-Object:")
print(verb_and_object(train_data))
print(verb_and_object(dev_data))
print(verb_and_object(test_data))

# 13. Distribution of gender-specific nouns in the dataset. Checked for those among animals, e.g. cow/bull, duck/drake, mare/stallion, but the dataset does not have those.

#List of gendered nouns to check against
masculine_nouns = ['man','father','boy','husband','uncle','policeman','dad','superman','son']
feminine_nouns = ['woman','mother','girl','wife','aunt','witch','girlfriend','grandma','women','daughter']
gender_neutral_nouns = ['person','parent','child','kid','baby','spouse','infant','people','human','student','chef','clown','baker','doctor','dentist','boxer','chef','officer','barber','wrestler']

print("Gendered nouns in train data:")
print(gendered_nouns(train_data))
print("Gendered nouns in dev data:")
print(gendered_nouns(dev_data))
print("Gendered nouns in test data:")
print(gendered_nouns(test_data))

# 15. word concreteness (wordnet, NLTK)
combined_vocab = train_unique_words.union(test_unique_words, dev_unique_words) # unions for sets
combined_tokens = train_tokens + test_tokens + dev_tokens # + for lists
print('15. Word concreteness using wordnet from NLTK:')
print('Terms:')
analyze_concreteness(combined_vocab)
print('Tokens:')
analyze_concreteness(combined_tokens)
