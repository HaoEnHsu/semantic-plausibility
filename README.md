Modeling Semantic Plausibility Project

Data Analysis:

Dataset: Pep3K

15 features to be analyzed:

Basic features of data:

1. size of dataset +
2. balance of dataset, how many data instances are labeled plausible versus implausible +
3. number of items in each data instance +
4. vocab in dataset CLARA
5. word count and token frequency CLARA

Linguistic Characteristics:

6. Part-of-Speech tag distribution SHAWN
   - intuition: though the distribution of POS tags in the dataset are mostly subject (Noun), verb, object (Noun), we found       some words are semantically ambiguous (e.g., the word chill can be both a noun, adjective, and verb); thus these words       may affect the model's performance
   - we used two different POS taggers: average perceptron tagger from NLTK (1/6 of the tokens are tagged as JJ) and       
     en_core_web_sm from spacy; and we chose to keep the results from spacy
   - library/packages used: pandas, matplotlib, spacy
   
7. how many times words appear in the dataset as subject or object and their corresponding labels SERGEI
8. (Dis)agreement in annotation, Kohen's kappa:
   each annotate selection of dataset and see if we agree with the labeled annotations +
9. running a kNN cluster against the dataset SERGEI

10. Relationships across data splits: CLARA
    compare basic features of data across test, train, and dev

11. word concreteness (wordnet, NLTK) CLARA
12. co-occurrence/collocation (compared to large corpus) SHAWN
    - intuition: if some n-grams (bigram and trigram here) occur frequently in a corpus, it is much more likely to be              plausible
    - corpus used: brown (W. N. Francis and H. Kucera [1964]) from NLTK (http://www.hit.uib.no/icame/brown/bcm.html)
    - library/packages used: pandas, nltk, matplotlib
      
13. action affordance (spaCy) SERGEI
14. commonsense knowledge (python package, conceptnet) CLARA
15. n-grams SHAWN
    
16. relation between subjects being animate/inanimate and their labels SHAWN
    - intuition: our hypothesis is that it is more likely that the sentences wtih animate subjects are plausible
    - dataset used: en-animacy-train (lingvenvist/en-animacy), link: https://huggingface.co/datasets/lingvenvist/en-               animacy/tree/main; after that we trimmed the dataset and keep only animate nouns (the entries labelled as "H" as human       and "A" as animate) and saved into another file called animate_nouns.txt
    - library/packages used: pandas, matplotlib
