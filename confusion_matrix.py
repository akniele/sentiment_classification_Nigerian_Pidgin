import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

datapath = f'./output_analysis_en.txt'
df = pd.read_csv(datapath, delimiter='\t')

true = df['true_label']
pred = df['pred_label']

plt.figure(figsize=(7, 5))

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# code found here: https://medium.com/mlearning-ai/confusion-matrix-for-multiclass-classification-f25ed7173e66
fx = sns.heatmap(confusion_matrix(true, pred), annot=True, fmt=".2f", cmap="GnBu")
fx.set_xlabel('\n Predicted Values\n')
fx.set_ylabel('Actual Values\n')
fx.xaxis.set_ticklabels(['negative', 'neutral', 'positive'])
fx.yaxis.set_ticklabels(['negative', 'neutral', 'positive'])
plt.tight_layout()
#plt.show()
plt.savefig("matrix")

# get recall, precision, etc.
print(classification_report(true, pred, digits=3))
