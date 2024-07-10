Shawn:


## Evaluation Results on Augmented Data

**K-means:**

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

- Dev data F1 Score: 0.6222 ; Accuracy: 0.6111 ; AUC-ROC score: 0.6583
- Test data F1 Score: 0.6144 ; Accuracy: 0.6156 ; AUC-ROC score: 0.6634
  
With combined animacy feature:

- Dev data F1 Score: 0.6076 ; Accuracy: 0.5948 ; AUC-ROC score: 0.6523
- Test data F1 Score: 0.6267 ; Accuracy: 0.6352 ; AUC-ROC score: 0.6742

Without animacy feature:

- Dev data F1 Score: 0.6149 ; Accuracy: 0.6111 ; AUC-ROC score: 0.6670
- Test data F1 Score: 0.6; Accuracy: 0.5961 ; AUC-ROC score: 0.6275
  

**BERT:**

With separated animacy feature:

- Dev data F1 Score: 0.7393 ; Accuracy: 0.7410 ; AUC-ROC score: 0.7700
- Test data F1 Score: 0.7123 ; Accuracy: 0.7255 ; AUC-ROC score: 0.8082

With combined animacy feature:

- Dev data F1 Score: 0.7395 ; Accuracy: 0.7344 ; AUC-ROC score: 0.7709
- Test data F1 Score: 0.7272 ; Accuracy: 0.7353 ; AUC-ROC score: 0.8085

Without animacy feature:

- Dev data F1 Score: 0.7355 ; Accuracy: 0.7311 ; AUC-ROC score: 0.7755
- Test data F1 Score: 0.7248 ; Accuracy: 0.7320 ; AUC-ROC score: 0.8106

  
## Project Setup

> :warning: **IMPORTANT: Please adjust the file paths/file names in the `BERT.py (Lines 10-13)`, `kmeans.py (Lines 28-31)` and `RandomForest.py (Lines 115-118)` files to the appropriate paths/names on your local machine before running the program.**

```python
python3 -m venv team_lab
source team_lab/bin/activate
requirement.txt
```
1. Python version:
   - Python (3.12.3)
     
2. Overview:
   
3. To get the models work, simply run the files.


