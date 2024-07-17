import spacy
import torchtext
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import googletrans
import random

# Load spacy model
spacy_en = spacy.load('en_core_web_sm')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset class
class TweetDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, vocab):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(tweet)
        indices = [self.vocab[token] for token in tokens]
        return torch.tensor(indices), torch.tensor(label)

# Load and clean data
tweetsDF = pd.read_csv("training.1600000.processed.noemoticon.csv", engine="python", header=None, encoding='ISO-8859-1')

tweetsDF["sentiment_cat"] = tweetsDF[0].astype('category')
tweetsDF["sentiment"] = tweetsDF["sentiment_cat"].cat.codes
tweetsDF.to_csv("train-processed.csv", header=None, index=None)
tweetsDF.sample(10000).to_csv("train-processed-sample.csv", header=None, index=None)

# Define tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# Build vocabulary
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

tweet_texts = tweetsDF[5].tolist()
vocab = build_vocab_from_iterator(yield_tokens(tweet_texts), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Create Dataset and DataLoaders
tweets = tweetsDF[5].tolist()
labels = tweetsDF["sentiment"].tolist()

dataset = TweetDataset(tweets, labels, tokenizer, vocab)
train_size = int(0.6 * len(dataset))
valid_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

def collate_batch(batch):
    tweet_list, label_list = [], []
    for (_tweet, _label) in batch:
        tweet_list.append(torch.tensor(_tweet, dtype=torch.int64))
        label_list.append(torch.tensor(_label, dtype=torch.int64))
    return pad_sequence(tweet_list, padding_value=vocab["<pad>"]), torch.tensor(label_list)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)

# Define LSTM model
class OurFirstLSTM(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size):
        super(OurFirstLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1)
        self.predictor = nn.Linear(hidden_size, 2)

    def forward(self, seq):
        output, (hidden, _) = self.encoder(self.embedding(seq))
        preds = self.predictor(hidden.squeeze(0))
        return preds

# Instantiate model
vocab_size = len(vocab)
model = OurFirstLSTM(100, 300, vocab_size)
model.to(device)

# Training
optimizer = optim.Adam(model.parameters(), lr=2e-2)
criterion = nn.CrossEntropyLoss()

def train(epochs, model, optimizer, criterion, train_loader, valid_loader):
    for epoch in range(1, epochs+1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch_idx, (tweets, labels) in enumerate(train_loader):
            tweets, labels = tweets.to(device), labels.to(device)
            optimizer.zero_grad()
            predict = model(tweets)
            loss = criterion(predict, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * len(labels)
        training_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            for batch_idx, (tweets, labels) in enumerate(valid_loader):
                tweets, labels = tweets.to(device), labels.to(device)
                predict = model(tweets)
                loss = criterion(predict, labels)
                valid_loss += loss.item() * len(labels)
        valid_loss /= len(valid_loader.dataset)
        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}'.format(epoch, training_loss, valid_loss))

train(5, model, optimizer, criterion, train_loader, valid_loader)

# Making predictions
def classify_tweet(tweet):
    categories = {0: "Negative", 1: "Positive"}
    model.eval()
    tokens = tokenizer(tweet)
    indices = torch.tensor([vocab[token] for token in tokens], dtype=torch.int64).unsqueeze(1).to(device)
    with torch.no_grad():
        output = model(indices)
    return categories[output.argmax().item()]

# Data Augmentation
import random
from random import randrange
from googletrans import Translator

def random_deletion(words, p=0.5):
    if len(words) == 1:
        return words
    remaining = list(filter(lambda x: random.uniform(0, 1) > p, words))
    return remaining if remaining else [random.choice(words)]

def random_swap(sentence, n=5):
    length = range(len(sentence))
    for _ in range(n):
        idx1, idx2 = random.sample(length, 2)
        sentence[idx1], sentence[idx2] = sentence[idx2], sentence[idx1]
    return sentence

# Note: you'll have to define remove_stopwords() and get_synonyms() elsewhere
def random_insertion(sentence, n):
    words = remove_stopwords(sentence)
    for _ in range(n):
        new_synonym = get_synonyms(random.choice(words))
        sentence.insert(randrange(len(sentence)+1), new_synonym)
    return sentence

translator = Translator()
sentences = ['The cat sat on the mat']
translations_fr = translator.translate(sentences, dest='fr')
fr_text = [t.text for t in translations_fr]
translations_en = translator.translate(fr_text, dest='en')
en_text = [t.text for t in translations_en]
print(en_text)

available_langs = list(googletrans.LANGUAGES.keys())
tr_lang = random.choice(available_langs)
print(f"Translating to {googletrans.LANGUAGES[tr_lang]}")

translations = translator.translate(sentences, dest=tr_lang)
t_text = [t.text for t in translations]
print(t_text)

translations_en_random = translator.translate(t_text, src=tr_lang, dest='en')
en_text = [t.text for t in translations_en_random]
print(en_text)
