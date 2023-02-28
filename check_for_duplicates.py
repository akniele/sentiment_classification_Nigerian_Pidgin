import pandas as pd

languages = ['en', 'pcm', 'ig', 'ha']
splits = ['train']

for lang in languages:
    for split in splits:
        df = pd.read_csv(f"{lang}_{split}.txt", sep="\t")
        df_no_duplicate_tweets = df.drop_duplicates(subset=['tweet'])
        df_no_duplicate_tweets.to_csv(f'pre_processed_{lang}_{split}_no_dup.txt', sep="\t", index=False)
