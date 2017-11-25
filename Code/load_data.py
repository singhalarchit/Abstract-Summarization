# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:56:24 2017

@author: Archit
"""

import cPickle as pickle
import nltk
import numpy as np
import re
import unicodedata
from io import open

full_data_filename = "../Processed Data/fulldata.pkl"
full_data_idx_filename = "../Processed Data/fulldataidx.pkl"
vocab_filename = "../Processed Data/vocab.pkl"

"""
def read_dataset2(filename):
    dataset = []
    with open(filename) as f:
        for line in f:
            line = line.decode('utf-8').encode('ascii','ignore')
            nodigit = ''.join([i for i in line if not i.isdigit()])
            dataset.append(nodigit)
    return dataset
"""

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

# Lowercase, trim, and remove non-letter characters
def normalizeString(s, noPeriod):
    s = unicodeToAscii(s.lower().strip())
    if noPeriod:
        s = re.sub(r"[^a-zA-Z]+", r" ", s)
    else:
        s = re.sub(r"([.])", r" \1 ", s)
        s = re.sub(r"[^a-zA-Z.]+", r" ", s)
    return s

def read_dataset(filename, noPeriod = False):
    dataset = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            dataset.append(str(normalizeString(line, noPeriod)))
    return dataset

def save_full_data():
    abstract_file = '../Processed Data/abstracts.txt'
    title_file = '../Processed Data/titles.txt'
    abstracts, titles = read_dataset(abstract_file), read_dataset(title_file, True)
    pickle.dump([abstracts, titles], open(full_data_filename, "wb"), \
                pickle.HIGHEST_PROTOCOL)

def load_full_data():
    abstracts, titles = pickle.load(open(full_data_filename, "rb"))
    return abstracts, titles

def abstracts2idx(abstracts):
    word2idx, _, _ = pickle.load(open(vocab_filename, "rb"))
    mod_abstracts = []
    for abstract in abstracts:
        sentences = nltk.sent_tokenize(abstract)[:10]
        mod_abstract = []
        for sentence in sentences:
            tokens = sentence.split()
            tokens2idx = [word2idx[token] for token in tokens]
            mod_abstract.append(tokens2idx)
        mod_abstracts.append(mod_abstract)
    return mod_abstracts

def titles2idx(titles):
    word2idx, _, _ = pickle.load(open(vocab_filename, "rb"))
    mod_titles = []
    for title in titles:
        tokens = title.split()
        tokens2idx = [word2idx[token] for token in tokens]
        mod_titles.append(tokens2idx)
    return mod_titles

def save_full_data_idx():
    abstracts, titles = load_full_data()
    abstracts, titles = abstracts2idx(abstracts), titles2idx(titles)
    pickle.dump([abstracts, titles], open(full_data_idx_filename, "wb"), \
                pickle.HIGHEST_PROTOCOL)
    
def load_full_data_idx():
    abstracts, titles = pickle.load(open(full_data_idx_filename, "rb"))
    return abstracts, titles

def split_data(abstracts, titles):
    np.random.seed(1)
    size = len(abstracts)
    ind = np.arange(size)
    np.random.shuffle(ind)
    abstracts = np.asarray(abstracts)[ind]
    titles = np.asarray(titles)[ind]
    abstracts_split = np.split(abstracts, [int(0.8*size), int(0.9*size), size])
    titles_split = np.split(titles, [int(0.8*size), int(0.9*size), size])
    return abstracts_split, titles_split

def get_splitted_data(idx = True):
    abstracts, titles = load_full_data()
    if idx:
        abstracts, titles = load_full_data_idx()
    else:
        abstracts, titles = load_full_data()
    abstracts_split, titles_split = split_data(abstracts, titles) #15874, 1984, 1985
    return abstracts_split[:3], titles_split[:3]    

#save_full_data()
#abstracts, titles = load_full_data()
#save_full_data_idx()
#abstracts, titles = load_full_data_idx()
#abstracts, titles = get_splitted_data()