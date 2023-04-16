import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt 
from sklearn.metrics import f1_score
from collections import defaultdict
import pickle
import sys
from acc_loss_plot import acc_loss_plot

EPOCHS = 4
LR = 5e-5
BATCHSIZE_TRAIN = 32
BATCHSIZE_TEST = 1

np.random.seed(112)
seed = sys.argv[1]  # specify seed (command line argument)
torch.manual_seed(int(seed))

pre = sys.argv[2]  # specify pre-training language ("mono" or "multi")

assert pre == "mono" or pre == "multi", "The accepted inputs for this argument are 'mono' and 'multi'!"

if pre == "mono":
    MODEL_TYPE = "bert-base-uncased"
else:
    MODEL_TYPE = "bert-base-multilingual-uncased"

fine = sys.argv[3]  # specify fine-tuning language(s)

languages = fine.split(",")

assert languages == ["pcm"] or languages == ["ig", "ha"] or languages == ["en"], \
    "The accepted inputs are 'pcm', 'ig,ha' and 'en'!"

# file name for saving model (file ending: .pt)
#model_name = f"{'_'.join(languages)}_final_{pre}_{seed}.pt"

# file names for saving some additional info (file ending: .pkl)
raw_data_file = f"raw_{'_'.join(languages)}_{pre}_{seed}.pkl"

# file name for the accuracy-loss plot
acc_loss_name = f"acc_loss_{'_'.join(languages)}_{pre}_{seed}.png"


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['label']]
        self.texts = [tokenizer(text,
                                padding='max_length', max_length=128, truncation=True,
                                return_tensors="pt") for text in df['tweet']]  # returns tokenized tweets
        self.raw_texts = [text for text in df['tweet']]  # returns the tweets themselves (human-readable)
        self.tweet_ids = [tweet_id for tweet_id in df['ID']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def get_raw_texts(self, idx):
        # Fetch a batch of raw input texts
        return self.raw_texts[idx]
    
    def get_tweet_ids(self, idx):
        # Fetch a batch of tweet ids
        return self.tweet_ids[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        raw = self.get_raw_texts(idx)
        ids = self.get_tweet_ids(idx)
        return batch_texts, batch_y, (raw, ids)


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(MODEL_TYPE)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 3)
        self.softmax = nn.LogSoftmax(dim=1)
        self.train_acc = list()
        self.train_loss = list()
        self.val_acc = list()
        self.val_loss = list()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.softmax(linear_output)

        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, BATCHSIZE_TRAIN, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, BATCHSIZE_TRAIN)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        model.train()

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label, (_, _) in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        model.train_acc.append(total_acc_train / len(train_data))
        model.train_loss.append(total_loss_train / len(train_data))
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            model.eval()

            for val_input, val_label, (_, _) in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

            model.val_acc.append(total_acc_val / len(val_data))
            model.val_loss.append(total_loss_val / len(val_data))

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .4f} | Train Accuracy:'
            f' {total_acc_train / len(train_data): .4f} | Val Loss: {total_loss_val / len(val_data): .4f}'
            f' | Val Accuracy: {total_acc_val / len(val_data): .4f}')

    acc_loss = defaultdict()
    acc_loss["train_accuracy"] = model.train_acc
    acc_loss["train_loss"] = model.train_loss
    acc_loss["val_accuracy"] = model.val_acc
    acc_loss["val_loss"] = model.val_loss
    
    acc_loss_plot(acc_loss, acc_loss_name)
    

def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, BATCHSIZE_TEST)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    y_true = list()
    y_pred = list()
    results_dict = defaultdict()
    
    with torch.no_grad():
        model.eval()

        for test_input, test_label, (raw_tweet, tweet_id) in test_dataloader:
            y_true.append(test_label.item())
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            y_pred.append(output.argmax(dim=1).item())
            results_dict[tweet_id] = (raw_tweet, test_label.item(), output.argmax(dim=1).item())

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    score_f1 = f1_score(y_true, y_pred, average='weighted')
    
    with open(raw_data_file, "wb") as p:
        pickle.dump(results_dict, p)

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f} \nF1 score: {score_f1}')


if __name__ == "__main__":

    # associate sentiments with numbers (used in Dataset class)
    labels = {'negative': 0,
              'neutral': 1,
              'positive': 2,
              }

    # prepare train and val data
    df_train = pd.DataFrame(columns=['ID', 'label', 'tweet'])
    df_val = pd.DataFrame(columns=['ID', 'label', 'tweet'])

    for language in languages:
        # read train, val, and test data into pandas df
        datapath_train = f'../data/pre_processed_{language}.txt'
        df_train_temp = pd.read_csv(datapath_train, delimiter='\t')

        datapath_val = f'../data/pre_processed_val_{language}.txt'
        df_val_temp = pd.read_csv(datapath_val, delimiter='\t')

        # shuffle
        df_train_temp = df_train_temp.sample(frac=1, random_state=42)
        df_val_temp = df_val_temp.sample(frac=1, random_state=42)

        df_train = pd.concat([df_train, df_train_temp], ignore_index=True)
        df_val = pd.concat([df_val, df_val_temp], ignore_index=True)

    # test data (pcm)
    datapath_test = f'../data/pre_processed_test_pcm.txt'
    df_test = pd.read_csv(datapath_test, delimiter='\t')

    tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)  # instantiate tokenizer
    model = BertClassifier()  # instantiate model

    train(model, df_train, df_val, LR, EPOCHS)  # train model
    torch.save(model.state_dict(), f'{model_name}')  # save model's state dict
    evaluate(model, df_test)  # evaluate model on test data
