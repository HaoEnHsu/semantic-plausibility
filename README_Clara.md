Clara:

Model: FFN

Directions:(it is assumed that the main directions on setting up a virtual environment and using the requirements.txt file have already been followed.)

To run:

Simply run the files. This will give you the evaluation on the already saved models.

To train:

If you would like to train a new model, go to the bottom of the files and uncomment the training model part, and comment out the section that loads the pretrained model.
Then simply run the model. To reduce training time, you can lower the number of epochs. It is set to 1250, because we found increased performance at that amount, but the model still does fine with less.

To train on the augmented data set:

Follow the directions labeled 1. in the files, start right at the end of the functions and continue through the code until you find the 1. that states it is the last one.

To retrain the BERT embeddings:

If you would like to run our BERT embeddings, go to the large section of commented out Bert embeddings and uncomment it. Look and make sure that you are running on the desired training data, there is one for augmented training data and the original training data.


Versions: 

ffn.py

Results:
Test Set: Accuracy: 0.6871
Dev Set: Accuracy: 0.6993
This is the baseline FFN. It has been trained without the extra anaimcy feature on the original data set. It can also be run on the augmented data, but performance goes down.

fnn_animacy.py

Results:
Test Set: Accuracy: 0.7068
Dev Set: Accuracy: 0.7026
This is the FNN with one animacy feature added. If an animate subject or object is present in the data instance, feature is 1, if more than 1, 2, if no animacy present 0.
This can also be trained on augmented data set but it decreases performance.

fnn_2animacy.py

Results:
Test Set: Accuracy: 0.6449
Dev Set: Accuracy: 0.6111
This is the FNN with two animacy features added. Two features added, one for object and one for subject. 1 if animate, 0 if inanimate.
This can also be trained on augmented data set but it decreases performance.



