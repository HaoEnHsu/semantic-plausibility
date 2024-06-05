Modeling Semantic Plausibility Project

Data Analysis:

Dataset: Pep3K

15 features to be analyzed:

Basic features of data:

1. Size of dataset.
2. Label balance of dataset, how many data instances are labeled plausible versus implausible.
3. Number of items in each data instance.
4. Vocabulary in dataset.
5. Word count and token frequency.

6. Out-of-vocabulary (OOV) words in dev and test data.
   Number of OOV words in dev set: 13
   Dev set OOV percentage: 0.039
   Number of OOV words in test set: 12
   Test set OOV percentage: 0.038

7. N-grams (bigrams and trigrams).

8. Relationships across data splits: CLARA
   compare basic features of data across test, train, and dev

Advanced features of data:

9. (Dis)agreement in annotation, Cohen's kappa. SERGEI
   We have randomly selected 20 instances and each of us annotated them to see if we agree with the labeled annotations. Cohen's kappa (imported from sklearn.metrics) was used for the calculation of the agreement in annotation. The results are as follows:
   Clara vs Gold kappa score: 0.9
   Sergei vs Gold kappa score: 0.61
   Hao-En vs Gold kappa score: 1.0
   Average kappa score: 0.83 (strong agreement)

10. Co-occurrence/collocation (compared to large corpus).

    - intuition: if some n-grams (bigram and trigram here) occur frequently in a corpus, it is much more likely to be plausible
    - corpus used: brown (W. N. Francis and H. Kucera [1964]) from NLTK (http://www.hit.uib.no/icame/brown/bcm.html)
    - library/packages used: pandas, nltk, matplotlib

11. Part-of-Speech tag distribution SHAWN

- intuition: though the distribution of POS tags in the dataset are mostly subject (Noun), verb, object (Noun), we found some words are semantically ambiguous (e.g., the word chill can be both a noun, adjective, and verb); thus these words may affect the model's performance
- we used two different POS taggers: average perceptron tagger from NLTK (1/6 of the tokens are tagged as JJ) and  
  en_core_web_sm from spacy; and we chose to keep the results from spacy
- library/packages used: pandas, matplotlib, spacy

12. Number of unique words that appear in the dataset as both subject and object (S-O), subject and verb (S-V), verb and object (V-O). SERGEI

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

    We have created three lists of gender-specific nouns (masculine, feminine, and neutral) that are common in English and appear in the dataset to see if the distribution is balanced. Nouns relating to human beings only were considered.

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

14. Relation between subjects being animate/inanimate and their labels SHAWN

    - intuition: our hypothesis is that it is more likely that the sentences wtih animate subjects are plausible
    - dataset used: en-animacy-train (lingvenvist/en-animacy), link: https://huggingface.co/datasets/lingvenvist/en-animacy/tree/main; after that we trimmed the dataset and keep only animate nouns (the entries labelled as "H" as human and "A" as animate) and saved into another file called animate_nouns.txt
    - library/packages used: pandas, matplotlib

15. Commonsense knowledge (python package, conceptnet).

16. Word concreteness (wordnet, NLTK).
