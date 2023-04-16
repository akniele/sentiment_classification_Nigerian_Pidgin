import csv
import pandas as pd
from sklearn.metrics import recall_score
import numpy as np


def get_recall_scores():
    languages = ["en", "pcm", "ig_ha"]

    pre_train = ["mono", "multi"]

    recall_dict = dict()

    for language in languages:
        for pre in pre_train:
            recalls = np.zeros((4, 3))  # four seeds, three classes (neg, neut, pos)
            for seed in range(2, 5+1):
                datapath = f"raw_{language}_{pre}_{seed}.txt"
                df = pd.read_csv(datapath, delimiter='\t')

                true = df['true_label']
                pred = df['pred_label']
                
                recalls[seed-2] = recall_score(true, pred, average=None)
      
            recall_dict[f"{language}_{pre}"] = np.mean(recalls, axis=0)
            
            
    with open('recall_scores.txt', 'w') as f:
        for (key,value) in recall_dict.items():
            f.write(f"{key}:\tnegative: {value[0]}, neutral: {value[1]}, positive: {value[2]}\n\n") 
			
			
if __name__ == "__main__":
    get_recall_scores()