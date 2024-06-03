import pandas as pd
import matplotlib.pyplot as plt

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
print("2. balance of dataset, how many data instances are labeled plausible versus implausible \nLabel counts:\n", label_counts_dict, '\n')
plt.figure(figsize=(8, 6)) # makes plot
label_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Counts of Each Label')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# 3. number of items in each data instance
word_counts = train_texts.apply(lambda x: len(x.split()))
word_length_counts = word_counts.value_counts().sort_index()
word_length_dict = word_length_counts.to_dict()
print("3. number of items in each data instance:\n", word_length_dict)
plt.figure(figsize=(10, 6)) # makes bar chart
word_length_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Word Counts in Text Instances')
plt.xlabel('Number of Words')
plt.ylabel('Number of Instances')
plt.xticks(rotation=0)
plt.show()
