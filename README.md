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
7. how many times words appear in the dataset as subject or object and their corresponding labels SERGEI
8. (Dis)agreement in annotation, Kohen's kappa:
   each annotate selection of dataset and see if we agree with the labeled annotations +
9. running a kNN cluster against the dataset SERGEI

10. Relationships across data splits: CLARA
    compare basic features of data across test, train, and dev

11. word concreteness (wordnet, NLTK) CLARA
12. co-occurrence/collocation (compared to large dataset) SHAWN
13. action affordance (spaCy) SERGEI
14. commonsense knowledge (python package, conceptnet) CLARA
15. n-grams SHAWN
16. relation between subjects being animate/inanimate and their labels SHAWN
