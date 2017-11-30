# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:54:26 2017

@author: Archit

Code adapted from https://github.com/A-Jacobson/CNN_Sentence_Classification/blob/master/WordVectors.ipynb
"""

import numpy as np
import pickle
from data_summary import load_vocab as load_vocab


def load_glove(path):
    """
    creates a dictionary mapping words to vectors from a file in glove format.
    """
    with open(path) as f:
        glove = {}
        for line in f.readlines():
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            glove[word] = vector
        return glove


def glove_embeddings(path, embedding_dim, word2idx):
    np.random.seed(1)
    found, found_words = 0, set()
    all_words = set(word2idx.keys())
    with open(path) as f:
        embeddings = np.random.randn(len(word2idx), embedding_dim)
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                found += 1
                found_words.add(word)
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector
        return embeddings, found, list(all_words - found_words)


def save_embeddings():
    word2idx, _, _, _ = load_vocab()
    dimensions = [50, 100, 200, 300]
    for dimension in dimensions:
        glove_path = '../glove/glove.6B.' + str(dimension) + 'd.txt'
        embedding_path = '../Processed Data/glove' + str(dimension) + '.pkl'
        embeddings, _, _ = glove_embeddings(glove_path, dimension, word2idx)
        pickle.dump(embeddings, open(embedding_path, "wb"), pickle.HIGHEST_PROTOCOL)


def load_embeddings(dimension):
    embedding_path = '../Processed Data/glove' + str(dimension) + '.pkl'
    return pickle.load(open(embedding_path, "rb"))
 
 
#save_embeddings()
#embeddings = load_embeddings(300)
