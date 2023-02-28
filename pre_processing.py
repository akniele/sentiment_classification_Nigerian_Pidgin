import csv
import re
import string
import emoji  # to deal with emojis
from pyarabic.araby import is_arabicrange  # to deal with arabic writing

unicode_emojis = ['♫', '♪', '・', '♬', '…', '☻', '•', '♡', '♔', '♚', '👳', 'ﷺ']  # unicode characters we don't want

languages = ['ig', 'ha', 'en', 'pcm']
splits = ['train', 'val', 'test']

for lang in languages:
    for split in splits:
        outfile = open(f"pre_processed_{split}_{lang}.txt", "w", encoding="utf-8")
        with open(f"{lang}_{split}.txt", encoding="utf-8") as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                tweet = row[1]
                cleaned_tweet = re.sub(r'http\S+', '', tweet)  # remove hyperlinks
                for symbol in unicode_emojis:
                    cleaned_tweet = re.sub(symbol, '', cleaned_tweet)  # removes the unicode symbols specified in list above
                cleaned_tweet = re.sub(r' *\@\S+ *', '', cleaned_tweet)  # removes mentions
                cleaned_tweet = cleaned_tweet.translate(str.maketrans('', '', string.punctuation))  # removes punctuation
                cleaned_tweet = cleaned_tweet.lower().strip(" ")
                cleaned_tweet = cleaned_tweet.split()
                cleaned_tweet = " ".join(cleaned_tweet)
                cleaned_tweet = emoji.replace_emoji(cleaned_tweet, replace='')

                encoded_tweet = cleaned_tweet.encode()  # encode tweet
                for word in cleaned_tweet:
                    if is_arabicrange(word):    # check if word is arabic
                        encoded_word = str.encode(word, 'utf-8')    # encode it cus otherwise it's not a string
                        encoded_tweet = re.sub(encoded_word, "".encode(), encoded_tweet)  # substitute arabic word for ""
                cleaned_tweet = encoded_tweet.decode()  # decode tweet again
                cleaned_tweet = cleaned_tweet.lower().strip(" ")
                cleaned_tweet = cleaned_tweet.split()  # split tweet (turns into list)
                cleaned_tweet = " ".join(cleaned_tweet)  # join list into string again (-> removes trailing white spaces)

                outfile.write(f"{row[0]}\t{row[2]}\t{cleaned_tweet}\n")  # write to new file

        outfile.close()
