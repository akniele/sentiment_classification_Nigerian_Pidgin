import pickle

languages = ["en", "pcm", "ig_ha"]

pre_train = ["mono", "multi"]

for language in languages:
    for pre in pre_train:
        for seed in range(2, 5+1):
            # write tweet ids, tweets, true and predicted labels to file
            outfile = open(f"raw_{language}_{pre}_{seed}.txt", "w", encoding="utf8")

            outfile.write("ID\ttweet\ttrue_label\tpred_label\n")

            with open(f"raw_{language}_{pre}_{seed}.pkl", "rb") as pickle_dic:
                dictionary = pickle.load(pickle_dic)

                for tweet_id, (raw, true, pred) in dictionary.items():
                    outfile.write(f"{tweet_id[0]}\t{raw[0]}\t{true}\t{pred}\n")
