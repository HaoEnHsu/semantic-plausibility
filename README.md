## Project Setup

> :warning: **IMPORTANT: Please adjust the file paths/file names in the `BERT.py (Lines 10-13)`, `kmeans.py (Lines 28-31)` and `RandomForest.py (Lines 115-118)` files to the appropriate paths/names on your local machine before running the program.`KNN`: follow the multistring instructions in knn.py to get the results for a desired model**

```python
python3 -m venv semantic-plausibility
source semantic-plausibility/bin/activate
pip install -r requirements.txt
```

1. Python version:
   - Python (3.12.3)
2. Overview:

Five models in total have been implemented: K-means, Random Forest, KNN, FNN, BERT.

**K-means**
**Random Forest**
**KNN**
Hyperparameters: k=9, distance='manhattan'. Other k numbers and distances have been tested, but this pair generally gives the best results.

The model's settings are preselected to showing the results for the original data and separated animacy features added, as it has the highest AUC-ROC score. If you want to see the results for other options, please follow the multistring instructions in knn.py.

**FNN**

To run:

Simply run the files. This will give you the evaluation on the already saved models.

To train:

If you would like to train a new model, go to the bottom of the files and uncomment the training model part, and comment out the section that loads the pretrained model.
Then simply run the model. To reduce training time, you can lower the number of epochs. It is set to 1250, because we found increased performance at that amount, but the model still does fine with less. 

To train on the augmented data set:

All files can also be trained on augmented data set but it decreases performance. Follow the directions labeled 1. in the files, start right at the end of the functions and continue through the code until you find the 1. that states it is the last one.

To retrain the BERT embeddings:

If you would like to run our BERT embeddings, go to the large section of commented out Bert embeddings and uncomment it. Look and make sure that you are running on the desired training data, there is one for augmented training data and the original training data.

With separated animacy feature: fnn_2animacy.py
- Test Set: Accuracy: 0.6449
- Dev Set: Accuracy: 0.6111

This is the FNN with two animacy features added. Two features added, one for object and one for subject. 1 if animate, 0 if inanimate.


With concatenated animacy feature: fnn_animacy.py
- Test Set: Accuracy: 0.7068
- Dev Set: Accuracy: 0.7026

This is the FNN with one animacy feature added. If an animate subject or object is present in the data instance, feature is 1, if more than 1, 2, if no animacy present 0.


Without animacy feature: fnn.py
- Test Set: Accuracy: 0.6871
- Dev Set: Accuracy: 0.6993

This is the baseline FFN. It has been trained without the extra anaimcy feature on the original data set. 


**BERT**
Tune the hyperparameters in lines 141-144

3. To get the models work, simply run the files.

## Evaluation Results on Augmented (Original) Data

**K-means (performances on augmented data and original data are identical)**

With separated animacy feature:

- Dev data F1 Score: 0.5784 ; Accuracy: 0.6046 ; AUC-ROC score: 0.6046
- Test data F1 Score: 0.5068 ; Accuracy: 0.5244 ; AUC-ROC score: 0.5243

With concatenated animacy feature:

- Dev data F1 Score: 0.5764 ; Accuracy: 0.6013 ; AUC-ROC score: 0.6013
- Test data F1 Score: 0.5068 ; Accuracy: 0.5244 ; AUC-ROC score: 0.5243

Without animacy feature:

- Dev data F1 Score: 0.5764 ; Accuracy: 0.6013 ; AUC-ROC score: 0.6013
- Test data F1 Score: 0.5390 ; Accuracy: 0.5375 ; AUC-ROC score: 0.5375

**Random Forest:**

With separated animacy feature:

- Dev data F1 Score: 0.6222 (0.6433); Accuracy: 0.6111 (0.6340); AUC-ROC score: 0.6583 (0.6728)
- Test data F1 Score: 0.6144 (0.5753); Accuracy: 0.6156 (0.5961); AUC-ROC score: 0.6634 (0.6532)

With combined animacy feature:

- Dev data F1 Score: 0.6076 (0.6026); Accuracy: 0.5948 (0.6078); AUC-ROC score: 0.6523 (0.6667)
- Test data F1 Score: 0.6267 (0.6367); Accuracy: 0.6352 (0.6319); AUC-ROC score: 0.6742 (0.6844)

Without animacy feature:

- Dev data F1 Score: 0.6149 (0.5933); Accuracy: 0.6111 (0.6013); AUC-ROC score: 0.6670 (0.6533)
- Test data F1 Score: 0.6 (0.6312); Accuracy: 0.5961 (0.6384); AUC-ROC score: 0.6275 (0.6833)

**KNN**

With separated animacy feature:

- Dev data F1 Score: 0.6075 (0.6105); Accuracy: 0.5947 (0.5915); AUC-ROC score: 0.6234 (0.6065)
- Test data F1 Score: 0.5855 (0.5825); Accuracy: 0.5895 (0.5798); AUC-ROC score: 0.6271 (0.6347)

With concatenated animacy feature:

- Dev data F1 Score: 0.5904 (0.6172); Accuracy: 0.5784 (0.5947); AUC-ROC score: 0.6006 (0.5926)
- Test data F1 Score: 0.5909 (0.5863); Accuracy: 0.5895 (0.5863); AUC-ROC score: 0.6341 (0.6282)

Without animacy feature:

- Dev data F1 Score: 0.5677 (0.5816); Accuracy: 0.5620 (0.5816); AUC-ROC score: 0.5835 (0.6068)
- Test data F1 Score: 0.5762 (0.5647); Accuracy: 0.5928 (0.5732); AUC-ROC score: 0.6188 (0.5940)

**BERT:**

With separated animacy feature:

- Dev data F1 Score: 0.7393 (0.7702); Accuracy: 0.7410 (0.7672); AUC-ROC score: 0.7700 (0.8103)
- Test data F1 Score: 0.7123 (0.7181); Accuracy: 0.7255 (0.7255); AUC-ROC score: 0.8082 (0.7735)

With combined animacy feature:

- Dev data F1 Score: 0.7395 (0.7453); Accuracy: 0.7344 (0.7311); AUC-ROC score: 0.7709 (0.8043)
- Test data F1 Score: 0.7272 (0.7059); Accuracy: 0.7353 (0.7059); AUC-ROC score: 0.8085 (0.7784)

Without animacy feature:

- Dev data F1 Score: 0.7355 (0.7538); Accuracy: 0.7311 (0.7344); AUC-ROC score: 0.7755 (0.8053)
- Test data F1 Score: 0.7248 (0.7101); Accuracy: 0.7320 (0.7092) ; AUC-ROC score: 0.8106 (0.7796)
