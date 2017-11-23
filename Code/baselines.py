# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:38:29 2017

@author: Archit
"""

import load_data as data
import nltk
import numpy as np
import rougescore as rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def similarity_baseline(abstracts):
    predictions, failed = [], []
    k = 0
    for abstract in abstracts:
        try:
            sentences = nltk.sent_tokenize(abstract)
            tfidf = TfidfVectorizer(stop_words='english')
            matrix = tfidf.fit_transform(sentences).toarray()
            similarity = cosine_similarity(matrix)
            most_similar = np.argmax(np.sum(similarity, 0))
            predictions.append(sentences[most_similar])
        except:
            predictions.append("")
            failed.append(k)
        print(k)
        k += 1
    return np.asarray(predictions), failed

def first_sent_baseline(abstracts):
    predictions, failed = [], []
    k = 0
    for abstract in abstracts:
        try:
            sentences = nltk.sent_tokenize(abstract)
            predictions.append(sentences[0])
        except:
            predictions.append("")
            failed.append(k)
        print(k)
        k += 1
    return np.asarray(predictions), failed

def informative_baseline(abstracts):
    predictions, failed = [], []
    k = 0
    for abstract in abstracts:
        try:
            sentences = nltk.sent_tokenize(abstract)
            tfidf = TfidfVectorizer(stop_words='english')
            matrix = tfidf.fit_transform(sentences).toarray()
            most_informative = np.argmax(np.sum(matrix, 1))
            predictions.append(sentences[most_informative])
        except:
            predictions.append("")
            failed.append(k)
        print(k)
        k += 1
    return np.asarray(predictions), failed

SET = 0
MODEL = 2

baselines = [similarity_baseline, first_sent_baseline, informative_baseline]
abstracts, titles = data.get_data()
titles_peer, failed = baselines[MODEL](abstracts[SET])
#titles_peer = abstracts[SET]
titles_model = titles[SET]
scores = rouge.eval_rouge(titles_peer, titles_model)
print(SET, MODEL)
print(scores)
print(failed)