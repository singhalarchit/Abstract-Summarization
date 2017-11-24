# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 00:59:22 2017

@author: Archit
"""

import cPickle as pickle
import nltk
from load_data import get_data
from rougescore import tokenize

FILENAME = "../Processed Data/vocab.pkl"

def full_data():
    abstracts, titles = get_data()
    abstracts = [abstract for typ in abstracts for abstract in typ]
    titles = [title for typ in titles for title in typ]
    return abstracts, titles

def get_summary(data):
    sent_len, seq_len, vocab = [], [], []
    for row in data:
        sentences = nltk.sent_tokenize(row)[:10]
        sent_len.append(len(sentences))
        len_tokens = 0
        for sentence in sentences:
            tokens = tokenize(sentence)
            len_tokens += len(tokens)
            vocab.extend(tokens)
        seq_len.append(len_tokens)
    return sent_len, seq_len, list(set(vocab))

def save_vocab():
    abstracts, titles = full_data()
    sent_len_a, seq_len_a, vocab_a = get_summary(abstracts)
    sent_len_t, seq_len_t, vocab_t = get_summary(titles)
    vocab = list(set(vocab_a + vocab_t))
    word2idx = {word: i for i, word in enumerate(vocab)}
    vocab_abstracts = [word2idx[word] for word in vocab_a]
    vocab_titles = [word2idx[word] for word in vocab_t]
    pickle.dump([word2idx, vocab_abstracts, vocab_titles], open(FILENAME, "wb"), \
                pickle.HIGHEST_PROTOCOL)

def get_vocab():
    try:
        word2idx, vocab_abstracts, vocab_titles = pickle.load(open(FILENAME, "rb"))
    except:
        save_vocab()
        word2idx, vocab_abstracts, vocab_titles = pickle.load(open(FILENAME, "rb"))
    return word2idx, vocab_abstracts, vocab_titles

word2idx, vocab_abstracts, vocab_titles = get_vocab()


"""
titles
sent_len
min: 1, max: 9, mean: 1.0172856926875977
[(1, 19540), (2, 280), (3, 11), (4, 11), (9, 1)]
seq_len
min: 1, max: 165, mean: 8.4750289774731637
cap to 25 words
14971 vocab size
14728 w/o numbers

BEFORE
abstracts
sent_len
min: 1, max: 705, mean = 11.188378773370962
[(4, 4065), (5, 3684), (3, 3076), (6, 2461), (7, 1475), (2, 1243), (8, 793), \
(9, 409), (1, 289), (10, 199), ...]
cap to 10 sentences
seq_len
min: 1, max: 12317, mean: 253.11202943103362
110640 vocab size

total 113574 unique vocab

AFTER
abstracts
sent_len
min: 1, max: 10, mean = 5.3055989517714055
[(4, 4065),  (5, 3684), (3, 3076),(6, 2461), (10, 2348), (7, 1475), (2, 1243), \
(8, 793), (9, 409), (1, 289)]
capped to 10 sentences
seq_len
min: 1, max: 5226, mean: 117.80713601773925
54496 vocab size
49835 w/o numbers

total 58020 unique vocab; 53297 w/o numbers
"""
