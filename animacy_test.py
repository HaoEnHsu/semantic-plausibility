import pandas as pd
import matplotlib.pyplot as plt

# Function to read the animate nouns list
def read_animate_nouns(filename):
  with open(filename, 'r', encoding='utf-8') as file:
    return set(line.strip() for line in file.readlines())

# Read the animate noun list
animate_nouns = read_animate_nouns('animate_nouns.txt')

# Function to process the data
def process_data(data_file):
  df = pd.read_csv(data_file)
  df['is_animate_noun'] = df['text'].apply(lambda text: text.split()[0] in animate_nouns)
  return df

# Process each data file
train_df = process_data('train.csv')
dev_df = process_data('dev.csv')
test_df = process_data('test.csv')
combined_df = pd.concat([train_df, dev_df, test_df])

# Plot the distribution of animacy labels
plt.figure(figsize=(8, 6))
combined_df['is_animate_noun'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title('Distribution of Animacy in Nouns (First Word in Text)')
plt.xlabel('Animacy Label (0=Not in List, 1=In List)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

print("Train Set Distribution:")
print(train_df['is_animate_noun'].value_counts())

print("\nDev Set Distribution:")
print(dev_df['is_animate_noun'].value_counts())

print("\nTest Set Distribution:")
print(test_df['is_animate_noun'].value_counts())
