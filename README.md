Shawn:


## Evaluation Results on Augmented Data

**Evaluation results for the Test and dev data with separated animacy feature:**
K-means:
- Dev data F1 Score: ; Accuracy: ; AUC-ROC score:
- Test data F1 Score: ; Accuracy: ; AUC-ROC score:
Random Forest: 
- Dev data F1 Score: ; Accuracy: ; AUC-ROC score:
- Test data F1 Score: ; Accuracy: ; AUC-ROC score:
BERT:
- Dev data F1 Score: ; Accuracy: ; AUC-ROC score:
- Test data F1 Score: ; Accuracy: ; AUC-ROC score:

**Evaluation results for the Test and dev data with concatenated animacy feature:**
K-means:
- Dev data F1 Score: ; Accuracy: ; AUC-ROC score:
- Test data F1 Score: ; Accuracy: ; AUC-ROC score:
Random Forest: 
- Dev data F1 Score: ; Accuracy: ; AUC-ROC score:
- Test data F1 Score: ; Accuracy: ; AUC-ROC score:
BERT:
- Dev data F1 Score: ; Accuracy: ; AUC-ROC score:
- Test data F1 Score: ; Accuracy: ; AUC-ROC score:

**Evaluation results for the Test and dev data without animacy feature:**
K-means:
- Dev data F1 Score: ; Accuracy: ; AUC-ROC score:
- Test data F1 Score: ; Accuracy: ; AUC-ROC score:
Random Forest: 
- Dev data F1 Score: ; Accuracy: ; AUC-ROC score:
- Test data F1 Score: ; Accuracy: ; AUC-ROC score:
BERT:
- Dev data F1 Score: ; Accuracy: ; AUC-ROC score:
- Test data F1 Score: ; Accuracy: ; AUC-ROC score:

  
## Project Setup

> :warning: **IMPORTANT: Please adjust the file paths/file names in the `BERT.py (Lines 10-13)`, `kmeans.py (Lines 17-20)` and `RandomForest.py (Lines 60-63)` files to the appropriate paths/names on your local machine before running the program.**

```python
python3 -m venv team_lab
source team_lab/bin/activate
requirement.txt
```
1. Python version:
   - Python (3.12.3)
     
2. Overview:
   
4. Run the following command to execute the program and get the evaluation results as well as the predictions for the appropriate data:
   `python3 main.py`


