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


# Vocabulary

vocabulary = set()
for i in train_data['text']:
    i = i.split()
    for j in i: 
        vocabulary.add(j)
# print(vocabulary)
print(len(vocabulary))

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
print(subject_object)
