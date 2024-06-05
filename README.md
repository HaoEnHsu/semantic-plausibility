Modeling Semantic Plausibility Project

Data Analysis

Dataset: Pep3K

Basic features of data:

To Start:
-have train, test, dev in same folder as .py code files
-run requirements.txt file

1. Size of dataset.

   - simply run the file, and output prints to terminal. Dataset is not that big.

   Number of data instances in train: 2449
   Number of data instances in testing data: 307
   Number of data instances in validation data: 306

2. Label balance of dataset, how many data instances are labeled plausible versus implausible.
   - see terminal outputs numbers and generated bar chart. The data is balanced.
     Train label counts:
     {0: 1227, 1: 1222}
     Test label counts:
     {0: 154, 1: 153}
     Validation label counts:
     {0: 153, 1: 153}
3. Number of items in each data instance.
   - see terminal output and generated bar chart. Each data instance has 3 words.
     {3: 2449}
4. Vocabulary in dataset.
   - see terminal output, the total number of unique terms in the dataset is smaller than we expected.
     Number of unique terms in the train dataset: 561
     Number of unique terms in the test dataset: 315
     Number of unique terms in the dev dataset: 330
5. Word count and token frequency.

   - see terminal output
     Number of tokens in the train dataset: 7347
     Number of tokens in the test dataset: 921
     Number of tokens in the dev dataset: 918

6. Out-of-vocabulary (OOV) words in dev and test data.

   - see terminal output, not many words are OOV.
     Number of OOV words in dev set: 13
     Dev set OOV percentage: 0.039
     Number of OOV words in test set: 12
     Test set OOV percentage: 0.038

7. N-grams (bigrams and trigrams).

   - number of unique train bigram: 3347
     number of unique train trigram: 2449
     number of unique test bigram: 587
     number of unique test trigram: 307
     number of unique dev bigram: 587
     number of unique dev trigram: 306
     
8. Relationships across data splits.
   - see terminal output and generated Zipf graphs
   - basic features of data across test, train, and dev are compared in 1-5, they are very similar. Zipf graph shows normal characteristics of Zipf's law

Advanced features of data:

9. (Dis)agreement in annotation, Cohen's kappa.
   library/packages used: sklearn
   We have randomly selected 20 instances and each of us annotated them to see if we agree with the labeled annotations. Cohen's kappa (imported from sklearn.metrics) was used for the calculation of the agreement in annotation. The results are as follows:
   Clara vs Gold kappa score: 0.9
   Sergei vs Gold kappa score: 0.61
   Hao-En vs Gold kappa score: 1.0
   Average kappa score: 0.83 (strong agreement)

   Looking at the dataset, we have found numerous instances, where it could be easy to disagree with the gold standard annotation.

10. Co-occurrence/collocation (compared to large corpus).

    - library/packages used: pandas, nltk, matplotlib
    - intuition: if some n-grams (bigram and trigram here) occur frequently in a corpus, it is much more likely to be plausible
    - corpus used: brown (W. N. Francis and H. Kucera [1964]) from NLTK (http://www.hit.uib.no/icame/brown/bcm.html)
    - collocation:
Bigram         Dataset Freq   Brown Corpus Freq
('open', 'door')        4       7
('open', 'window')      4       5
('cool', 'air')         2       2
('cool', 'water')       3       2
('warm', 'water')       2       2
('plane', 'crash')      1       2
('open', 'bottle')      2       1
('clean', 'slate')      1       1
('drink', 'water')      1       1
('sun', 'heat')         2       1
('drink', 'beer')       3       1
('cook', 'stove')       1       1
('air', 'cool')         2       1
('clean', 'house')      3       1
('oil', 'heat')         2       1
('horse', 'kick')       3       1
('pour', 'water')       2       1
('dog', 'shake')        3       0
('shake', 'cat')        1       0
('student', 'climb')    2       0


11. Part-of-Speech tag distribution.

    - library/packages used: pandas, matplotlib, spacy
    - intuition: though the distribution of POS tags in the dataset are mostly subject (Noun), verb, object (Noun), we found some words are semantically ambiguous (e.g., the word chill can be both a noun, adjective, and verb); thus these words may affect the model's performance
    - we used two different POS taggers: average perceptron tagger from NLTK (1/6 of the tokens are tagged as JJ) and en_core_web_sm from spacy; and we chose to keep the results from spacy
    - Train POS Tag Distribution: Counter({'NOUN': 4720, 'PROPN': 1596, 'VERB': 830, 'ADJ': 178, 'ADV': 11, 'AUX': 5, 'ADP': 4, 'INTJ': 2, 'PRON': 1})
    - Test POS Tag Distribution:({'NOUN': 597, 'PROPN': 198, 'VERB': 101, 'ADJ': 22, 'INTJ': 2, 'AUX': 1})
    - Dev POS Tag Distribution:({'NOUN': 589, 'PROPN': 214, 'VERB': 96, 'ADJ': 18, 'ADP': 1})


12. Number of unique words that appear in the dataset as both subject and object (S-O), subject and verb (S-V), verb and object (V-O). The terminal output is as follows:

    S-O Train: 157
    S-O Dev: 61
    S-O Test: 62

    S-V Train: 4
    S-V Dev: 0
    S-V Test: 1

    V-O Train: 1
    V-O Dev: 1
    V-O Test: 0

13. Distribution of gender-specific nouns in the dataset.

    We have created three lists of gender-specific nouns (masculine, feminine, and neutral) that are common in English and appear in the dataset to see if the distribution is balanced. Nouns relating to human beings only were considered, those from the animal domain, e.g. cow/bull, duck/drake, mare/stallion, were not found in the dataset, therefore this distinction is ignored.

    We have observed a gender imbalance in the dataset with a greater number of masculine nouns than feminine.

    Train set:
    Number of masculine nouns: 273, Percentage among gender-specific nouns: 0.35
    Number of feminine nouns: 199, Percentage among gender-specific nouns: 0.25
    Number of neutral nouns: 319, Percentage among gender-specific nouns: 0.40

    Dev set:
    Number of masculine nouns: 33, Percentage among gender-specific nouns: 0.31
    Number of feminine nouns: 29, Percentage among gender-specific nouns: 0.27
    Number of neutral nouns: 44, Percentage among gender-specific nouns: 0.42

    Test set:
    Number of masculine nouns: 38, Percentage among gender-specific nouns: 0.32
    Number of feminine nouns: 30, Percentage among gender-specific nouns: 0.25
    Number of neutral nouns: 52, Percentage among gender-specific nouns: 0.43

14. Relation between subjects being animate/inanimate and their labels.

    - intuition: our hypothesis is that it is more likely that the sentences wtih animate subjects are plausible
    - dataset used: en-animacy-train (lingvenvist/en-animacy), link: https://huggingface.co/datasets/lingvenvist/en-animacy/tree/main; after that we trimmed the dataset and keep only animate nouns (the entries labelled as "H" as human and "A" as animate) and saved into another file called animate_nouns.txt
    - library/packages used: pandas, matplotlib
    - Train Set Distribution: True: 1521 / False 928
    - Dev Set Distribution:   True: 178 / False 128
    - Test Set Distribution:  True  189 / False 118


15. Commonsense knowledge (python package, conceptnet).

    - go to the concept_net.py file
    - not many subject-object pairs are connected in the ConceptNet knowledge graph. Tested by any relation, not just the relation in the data. This makes sense because many objects/subjects in our dataset are not often seen together/do not have a typical relation to each other.
      Commonsense knowledge evaluation using ConceptNet:
      Number of subject-object pairs in dataset related by any relation in ConceptNet: 227
      Number of subject-object pairs in dataset NOT related by any relation in ConceptNet: 2222

16. Word concreteness (wordnet, NLTK).

    - wordnet should be downloaded in main file just by using requirements.txt and the code in file
    - see terminal
    - wordnet is a lexical database that contains synsets, or sets of synonyms, hypernyms, and hyponyms. This allows analysis of abstractness/concreteness using these relations in different ways. Our code analyzes this in three different ways: by classifying adjectives as abstract, defining a set of keywords that id they are in the def of example of a word's synset, the word is classified as abstract, and by defining a set of hypernyms(words more generic than subordinate), and if hypernym is in word's synset, word is classified as abstract. These are mostly reasonable rules, but our dataset is not ideal for these rules or for using wordnet in general because our dataset is so concrete. When looking at the dataset, it is difficult to find any abstract words, especially taking context into account. wordnet doesn't take context into account, if a word can be abstract in any possible context it marks it as abstract. By analyzing this feature we learned that wordnet is maybe not ideaL for our dataset and that our dataset has very concrete words.

    Word concreteness using wordnet from NLTK:
    Terms:
    Words that can be abstract: 27.23%
    Concrete words: 72.77%
    Words not in WordNet: 0.00%
    Tokens:
    Words that can be abstract: 26.64%
    Concrete words: 73.36%
    Words not in WordNet: 0.00%
