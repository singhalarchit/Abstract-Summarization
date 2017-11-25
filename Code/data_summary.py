# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 00:59:22 2017

@author: Archit
"""

import cPickle as pickle
import nltk
from load_data import load_full_data

vocab_filename = "../Processed Data/vocab.pkl"

"""
def full_data():
    abstracts, titles = get_data()
    abstracts = [abstract for typ in abstracts for abstract in typ]
    titles = [title for typ in titles for title in typ]
    return abstracts, titles

#import re
def tokenize(row):
    tokens = re.split('\W+', row.lower())
    tokens = [token for token in tokens if token != '']
    return tokens
"""

def get_summary(data):
    sent_len, seq_len, vocab = [], [], []
    for row in data:
        sentences = nltk.sent_tokenize(row)[:10]
        sent_len.append(len(sentences))
        len_tokens = 0
        for sentence in sentences:
            tokens = sentence.split()
            len_tokens += len(tokens)
            vocab.extend(tokens)
        seq_len.append(len_tokens)
    return sent_len, seq_len, list(set(vocab))

def save_vocab():
    abstracts, titles = load_full_data()
    sent_len_a, seq_len_a, vocab_a = get_summary(abstracts)
    sent_len_t, seq_len_t, vocab_t = get_summary(titles)
    vocab_t.append('<SOS>')
    vocab_t.append('<EOS>')
    vocab = list(set(vocab_a + vocab_t))
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for i, word in enumerate(vocab)}
    word2idx2titleidx = {word2idx[word]: i for i, word in enumerate(vocab_t)}
    pickle.dump([word2idx, idx2word, word2idx2titleidx], open(vocab_filename, "wb"), \
                pickle.HIGHEST_PROTOCOL)

def load_vocab():
    word2idx, idx2word, word2idx2titleidx = pickle.load(open(vocab_filename, "rb"))
    return word2idx, idx2word, word2idx2titleidx

#save_vocab()
#word2idx, idx2word, word2idx2titleidx = load_vocab()
