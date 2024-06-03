from sklearn.metrics import cohen_kappa_score
import pandas as pd

annotation_data = pd.read_csv("annotation agreement.csv")
# print(annotation_data)

clara_anot = annotation_data['Clara']
sergei_anot = annotation_data['Sergei']
shawn_anot = annotation_data['Shawn']
gold_standard = annotation_data['gold_standard']

clara_gold = (cohen_kappa_score(clara_anot,gold_standard))
sergei_gold = (cohen_kappa_score(sergei_anot,gold_standard))
shawn_gold = (cohen_kappa_score(shawn_anot,gold_standard))

aver_kappa_score = (clara_gold + sergei_gold + shawn_gold)/3
print(aver_kappa_score)
