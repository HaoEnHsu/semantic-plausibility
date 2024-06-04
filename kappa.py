from sklearn.metrics import cohen_kappa_score
import pandas as pd

train_data = pd.read_csv('train.csv')
# Kappa score for the train data
annotation_data = pd.read_csv("annotation agreement.csv")
# print(annotation_data)

clara_anot = annotation_data['Clara']
sergei_anot = annotation_data['Sergei']
shawn_anot = annotation_data['Shawn']
gold_standard = annotation_data['gold_standard']

clara_gold = (cohen_kappa_score(clara_anot,gold_standard))
sergei_gold = (cohen_kappa_score(sergei_anot,gold_standard))
shawn_gold = (cohen_kappa_score(shawn_anot,gold_standard))

aver_kappa_score = (clara_gold + sergei_gold + shawn_gold)/3
# print(aver_kappa_score)

# Total number of all words in train data

all_words_train = list()
for i in train_data['text']:
    i = i.split()
    for j in i:
        all_words_train.append(j)

# Train vocabulary

vocabulary_train = set()
for i in train_data['text']:
    i = i.split()
    for j in i: 
        vocabulary_train.add(j)
# print(vocabulary)

# Writing all unique words from train data into a new file to check for animacy

# animacy_file = 'animacy.txt'


# with open(animacy_file, 'w') as file:
#     for unique_word in vocabulary_train:
#         file.write(unique_word + '\n')



# Number of times a word is both S and O

instances = train_data['text'] 
subject = []
verb = [] 
object = []
for instance in instances:
    instance = instance.split()
    subject.append(instance[0])
    verb.append(instance[1])
    object.append(instance[2])

subject_object = []
for i in subject:
    if i in object:
        subject_object.append(i)
# print(subject_object)

''' Distribution of gender-specific nouns in the dataset.
Checked for those among animals, e.g. cow/bull, duck/drake, mare/stallion, but the dataset does not have those'''

# List of gendered nouns to check against
masculine_nouns = ['man','father','boy','husband','uncle','actor','policeman','dad']
feminine_nouns = ['woman','mother','girl','wife','aunt','actress','witch','girlfriend','grandma','women']
gender_neutral_nouns = ['person','parent','child','kid','baby','spouse','infant','people']

# Counting the number of occurences of M, F, and GN nouns
masculine_count = 0
feminine_count = 0
gender_neutral_count = 0

for i in all_words_train:
    if i in masculine_nouns:
        masculine_count += 1
    elif i in feminine_nouns:
        feminine_count += 1
    elif i in gender_neutral_nouns:
        gender_neutral_count += 1

# Overall number of gendered nouns. To be used for normalisation
total_sum = masculine_count + feminine_count + gender_neutral_count

# print(all_words)
# print("Masculine nouns:", masculine_count, ",Percentage:", masculine_count/total_sum)
# print("Feminine nouns:", feminine_count, ",Percentage:", feminine_count/total_sum)
# print("Gender neutral nouns:", gender_neutral_count, ",Percentage:", gender_neutral_count/total_sum)
