from sklearn.metrics import cohen_kappa_score
import pandas as pd

# reading from train, dev, test
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
dev_data = pd.read_csv('dev.csv')

# 8. Kappa score for the train data
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
# print(clara_gold)
# print(sergei_gold)
# print(shawn_gold)
# print(aver_kappa_score)

# Total number of all words 

def all_words(pd_file):
    list_of_all_words = list()
    for i in pd_file['text']:
        i = i.split()
        for j in i:
            list_of_all_words.append(j)
    return list_of_all_words

# Function extracting a set of unique words from a file

def vocabulary(pd_file):
    unique_words = set()
    for i in pd_file['text']:
        i = i.split()
        for j in i: 
           unique_words.add(j)
    return unique_words


# 7. Number of times a noun is both subject and object

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

# print("Subject-Object:")
# print(subject_and_object(train_data))
# print(subject_and_object(dev_data))
# print(subject_and_object(test_data))
# print("Subject-Verb:")
# print(subject_and_verb(train_data))
# print(subject_and_verb(dev_data))
# print(subject_and_verb(test_data))
# print("Verb-Object:")
# print(verb_and_object(train_data))
# print(verb_and_object(dev_data))
# print(verb_and_object(test_data))


''' 13. Distribution of gender-specific nouns in the dataset.
Checked for those among animals, e.g. cow/bull, duck/drake, mare/stallion, but the dataset does not have those'''

#List of gendered nouns to check against
masculine_nouns = ['man','father','boy','husband','uncle','policeman','dad','superman','son']
feminine_nouns = ['woman','mother','girl','wife','aunt','witch','girlfriend','grandma','women','daughter']
gender_neutral_nouns = ['person','parent','child','kid','baby','spouse','infant','people','human','student','chef','clown','baker','doctor','dentist','boxer','chef','officer','barber','wrestler']

# Counting the number of occurences of masculine, feminine, and neutral nouns
masculine_count = 0
feminine_count = 0
gender_neutral_count = 0

# For the statistics of gendered nouns, use train_data, dev_data, or test_data in the brackets below.
for i in all_words(test_data):
    if i in masculine_nouns:
        masculine_count += 1
    elif i in feminine_nouns:
        feminine_count += 1
    elif i in gender_neutral_nouns:
        gender_neutral_count += 1

# Overall number of gendered nouns. To be used for calculating the percentage.
total_sum = masculine_count + feminine_count + gender_neutral_count

# print("Number of masculine nouns:", masculine_count, ", Percentage among gender-specific nouns:", masculine_count/total_sum)
# print("Number of feminine nouns:", feminine_count, ", Percentage among gender-specific nouns:", feminine_count/total_sum)
# print("Number of neutral nouns:", gender_neutral_count, ", Percentage among gender-specific nouns:", gender_neutral_count/total_sum)


# 9. Out-of-vocabulary words in dev and test

vocabulary_train = vocabulary(train_data)
vocabulary_dev = vocabulary(dev_data)
vocabulary_test = vocabulary(test_data)

def out_of_vocabulary(vocabulary):
    out_of_vocab = set()
    for word in vocabulary:
        if word not in vocabulary_train:
            out_of_vocab.add(word)
    return out_of_vocab

oov_dev = out_of_vocabulary(vocabulary_dev)
oov_test = out_of_vocabulary(vocabulary_test)

print("Words in dev set that do not appear in train set: ",oov_dev)
print("Number of OOV words in dev set:",len(oov_dev), "\nDev OOV Percentage:",len(oov_dev)/len(vocabulary_dev)) #Output: 13, 0.039
print("Words in test set that do not appear in train set:", oov_test)
print("Number of OOV words in test set:",len(oov_test), "\nTest OOV Percentage:",len(oov_test)/len(vocabulary_test)) #Output: 12, 0.038

# print(len(vocabulary_dev))
# print(len(vocabulary_test))
