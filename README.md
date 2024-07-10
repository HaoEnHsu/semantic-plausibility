Shawn:


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
  

**BERT:**
**Tune the hyperparameters in lines 141-144

With separated animacy feature:

- Dev data F1 Score: 0.7393 (0.7702); Accuracy: 0.7410 (0.7672); AUC-ROC score: 0.7700 (0.8103)
- Test data F1 Score: 0.7123 (0.7181); Accuracy: 0.7255 (0.7255); AUC-ROC score: 0.8082 (0.7735)

With combined animacy feature:

- Dev data F1 Score: 0.7395 (0.7453); Accuracy: 0.7344 (0.7311); AUC-ROC score: 0.7709 (0.8043)
- Test data F1 Score: 0.7272 (0.7059); Accuracy: 0.7353 (0.7059); AUC-ROC score: 0.8085 (0.7784)

Without animacy feature:

- Dev data F1 Score: 0.7355 (0.7538); Accuracy: 0.7311 (0.7344); AUC-ROC score: 0.7755 (0.8053)
- Test data F1 Score: 0.7248 (0.7101); Accuracy: 0.7320 (0.7092) ; AUC-ROC score: 0.8106 (0.7796)

  
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


