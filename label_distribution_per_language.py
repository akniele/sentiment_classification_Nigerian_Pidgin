import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

languages = ['pcm', 'en', 'ig', 'ha']  # don't change this!

# lists for saving how many neutral, positive and negative tweets there are per language
neutral = []
positive = []
negative = []

for language in languages:
    datapath_train = f'../dev_and_test_data_gold_labels/pre_processed_files/pre_processed_{language}.txt'
    df = pd.read_csv(datapath_train, delimiter='\t')

    datapath_val = f'../dev_and_test_data_gold_labels/pre_processed_files/pre_processed_val_{language}.txt'
    df_val = pd.read_csv(datapath_val, delimiter='\t')
    df = pd.concat([df, df_val])  # concatenate val

    # get number of tweets per label
    neutral.append(df['label'].value_counts()['neutral'])
    positive.append(df['label'].value_counts()['positive'])
    negative.append(df['label'].value_counts()['negative'])

# add up number of tweets for ig and ha
neutral[2] = neutral[2] + neutral[3]
neutral = neutral[:-1]
positive[2] = positive[2] + positive[3]
positive = positive[:-1]
negative[2] = negative[2] + negative[3]
negative = negative[:-1]

# this is where the visualization part starts
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

labels = ['pcm', 'en', 'ig+ha']

x = np.arange(len(labels))/3  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, neutral, width, label='neutral')
rects2 = ax.bar(x, positive, width, label='positive')
rects3 = ax.bar(x + width, negative, width, label='negative')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of tweets')
ax.set_title('Number of tweets in the test set by sentiment')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

plt.ylim(0, 14000)
fig.tight_layout()

plt.savefig("label_distribution.png")
