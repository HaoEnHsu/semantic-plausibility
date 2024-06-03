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
print("Number of data instances in train:", num_instances)
num_instances = test_data.shape[0]
print("Number of data instances in testing data:", num_instances)
num_instances = dev_data.shape[0]
print("Number of data instances in validation data:", num_instances)

# 2. balance of dataset, how many data instances are labeled plausible versus implausible
label_counts = train_labels.value_counts()
print("Label counts:\n", label_counts)
plt.figure(figsize=(8, 6))
label_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Counts of Each Label')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# 3. number of items in each data instance

