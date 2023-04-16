import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.texts = [tokenizer(text,
                                padding='max_length', max_length=128, truncation=True,
                                return_tensors="pt") for text in df['tweet']]
        self.raw_texts = [text for text in df['tweet']]
        self.tweet_ids = [tweet_id for tweet_id in df['ID']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.tweet_ids)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def get_tweet_ids(self, idx):
        # Fetch a batch of tweet ids
        return self.tweet_ids[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        ids = self.get_tweet_ids(idx)
        return batch_texts, ids


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 3)
        self.relu = nn.LogSoftmax(dim=1)
        self.train_acc = list()
        self.train_loss = list()
        self.val_acc = list()
        self.val_loss = list()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    y_pred = list()
    results_dict = defaultdict()

    with torch.no_grad():

        for test_input, tweet_id in test_dataloader:
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            y_pred.append(output.argmax(dim=1).item())
            results_dict[tweet_id] = output.argmax(dim=1).item()

    with open("test_result_pcm_full_data.pkl", "wb") as p:
        pickle.dump(results_dict, p)


datapath = f"./pcm_test_participants.tsv"
df_test = pd.read_csv(datapath, delimiter='\t')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
labels = {'negative': 0,
          'neutral': 1,
          'positive': 2,
          }

model = BertClassifier()
model.load_state_dict(torch.load("../model/pcm_full_data.pt"))
model.eval()

evaluate(model, df_test)
