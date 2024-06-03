import pandas as pd # allows dataframes
# imports conceptnet, the file takes a long time to download
import conceptnet_lite
from conceptnet_lite import Label, edges_between
conceptnet_lite.connect("/path/to/conceptnet.db")

def test_commonsense(data):
    triples = [line.split() for line in data['text']] # extracts a list of dataset triples
    results = []
    for triple in triples:
        triple_result = False  # Initialize result for current triple
        concept_1 = Label.get(text=triple[0], language='en').concepts # subject
        concept_3 = Label.get(text=triple[2], language='en').concepts # object
        for e in edges_between(concept_1, concept_3, two_way=True): # allows relation to point either way
            #print("  Edge URI:", e.uri)
            #print("  Edge name:", e.relation.name)
            #print("  Edge start node:", e.start.text)
            #print("  Edge end node:", e.end.text)
            #print("  Edge metadata:", e.etc)
            if e.relation:
                triple_result = True
        results.append(triple_result) # list of whether there are relations between or not
        true_count = 0
        false_count = 0
        for result in results:
            if result == True:
                true_count += 1
            else:
                false_count += 1
    return true_count,false_count

#load files
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
dev_data = pd.read_csv('dev.csv')



# 14. commonsense knowledge (python package, conceptnet)
print('\n14. Commonsense knowledge evaluation using ConceptNet:')
true, false = test_commonsense(train_data)
print('Number of subject-object pairs in dataset related by any relation in ConceptNet:', true)
print('Number of subject-object pairs in dataset NOT related by any relation in ConceptNet:', false)
# this count is between the first and last word in each data instance, the 2nd word is considered a relation, and because so few matched, allowed that relation to be anything
