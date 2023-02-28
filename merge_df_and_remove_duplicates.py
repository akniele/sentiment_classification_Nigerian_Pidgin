import pandas as pd

filename1 = "en_dev1.txt"
filename2 = "en_dev2.txt"

df = pd.read_csv(filename1, sep="\t")

df_2 = pd.read_csv(filename2, sep="\t")

df_3 = pd.concat([df, df_2], axis=0, ignore_index=True)

print(f"neutral: {df_3['label'].value_counts()['neutral']}")
print(f"positive: {df_3['label'].value_counts()['positive']}")
print(f"negative: {df_3['label'].value_counts()['negative']}")

df_4 = df_3.drop_duplicates(subset=['tweet'])

df_4.to_csv(f"en_dev.txt", sep="\t", index=False)

print(f"neutral: {df_4['label'].value_counts()['neutral']}")
print(f"positive: {df_4['label'].value_counts()['positive']}")
print(f"negative: {df_4['label'].value_counts()['negative']}")
