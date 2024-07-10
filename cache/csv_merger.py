import pandas as pd

valid = pd.read_csv('new_valid.csv')
new_test = pd.read_csv('new_test.csv')
train = pd.read_csv('train.csv')

# Remove the last two columns to match the format the original pep3k
valid = valid.iloc[:,:-2]
new_test = new_test.iloc[:,:-2]

# Add columns to datasets
valid.columns = ['label','text1','text2','text3']
new_test.columns = ['label','text1','text2','text3']

# Merge three columns into one
def merge_columns(input_file, output_file, columns_to_merge, new_column_name):
    df = input_file
    
    # Merge columns into one by concatenating their string representations
    df[new_column_name] = df[columns_to_merge].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
    # Remove the now empty columns
    df.drop(columns=columns_to_merge, inplace=True)
    
    # Save the modified DataFrame back to a CSV file
    df.to_csv(output_file, index=False)


# input_file = valid
# output_file = 'data_augmented.csv'

input_file = new_test
output_file = 'temp_data_aug.csv'
columns_to_merge = ['text1','text2','text3']
new_column_name = 'text'

train_add = 'train.csv'
# merge_columns(input_file, output_file, columns_to_merge, new_column_name)


# # with open('train.csv','r') as train_add:
# with open('temp_data_aug.csv','r') as new_test_file:
#     with open('data_augmented.csv','a') as merged_file:
#         for i in new_test_file:
#         # for i in train_add:
#             merged_file.write(i)

train_augmented_data = pd.read_csv('data_augmented.csv')

# print(len(train_augmented_data['label'].values))