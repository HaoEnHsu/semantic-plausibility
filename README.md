Modeling Semantic Plausibility Project

Data Analysis:

Dataset: Pep3K

15 features to be analyzed:

Basic features of data:

1. size of dataset
2. balance of dataset, how many data instances are labeled plausible versus implausible
3. number of items in each data instance
4. vocab in dataset
5. word count and token frequency
6. average word length

Linguistic Characteristics:

7. Part-of-Speech tag distribution
8. relation between subjects being animate/inanimate and their labels
9. relation between objects being animate/inanimate and their labels
10. how many times words appear in the dataset as subject or object and their corresponding labels
11. (Dis)agreement in annotation:

12. each annotate selection of dataset and see if we agree with the labeled annotations
13. Relationships across data splits:

14. compare basic features of data across test, train, and dev
15. object-verb relationship (dependency parsing, spaCy)
16. word concreteness (wordnet, NLTK)
17. co-occurrence/collocation (compared to large dataset)
18. action affordance (spaCy)
19. commonsense knowledge (python package, conceptnet)
20. n-grams

16.
