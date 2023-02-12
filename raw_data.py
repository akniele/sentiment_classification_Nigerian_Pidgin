import pickle

language = "en"

# write tweet ids, tweets, true and predicted labels to file
outfile = open(f"raw_{language}.txt", "w", encoding="utf8")

outfile.write("ID\ttweet\ttrue_label\tpred_label\n")

with open(f"raw_{language}.pkl", "rb") as pickle_dic:
    dictionary = pickle.load(pickle_dic)

    for tweet_id, (raw, true, pred) in dictionary.items():
        outfile.write(f"{tweet_id[0]}\t{raw[0]}\t{true}\t{pred}\n")
