# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 17:36:13 2017

@author: Archit
"""

import itertools
import load_data as data
import networkx as nx
import nltk
import numpy as np
import rougescore as rouge
import warnings
from math import log10 as log
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")

def levenshtein_distance(first, second):
    """Return the Levenshtein distance between two strings.
    Based on:
        http://rosettacode.org/wiki/Levenshtein_distance#Python
    """
    if len(first) > len(second):
        first, second = second, first
    distances = range(len(first) + 1)
    for index2, char2 in enumerate(second):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(first):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1],
                                             distances[index1 + 1],
                                             new_distances[-1])))
        distances = new_distances
    return distances[-1]

def similarity_matrix(sentences):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(sentences).toarray()
    similarity = cosine_similarity(matrix)
    return similarity

def overlap(first, second):
    first = rouge.tokenize(first)
    second = rouge.tokenize(second)
    first_counter = rouge._ngram_counts(first, 1)
    second_counter = rouge._ngram_counts(second, 1)
    matches = rouge._counter_overlap(first_counter, second_counter)
    return matches/(log(len(first)) + log(len(second)))

def lcs(first, second):
    first = rouge.tokenize(first)
    second = rouge.tokenize(second)
    matches = rouge.lcs(first, second)
    return matches/(log(len(first)) + log(len(second)))

def build_graph(sentences):
    """Return a networkx graph instance.
    :param nodes: List of hashables that represent the nodes of a graph.
    """
    gr = nx.Graph()             # initialize an undirected graph
    nodes = np.arange(len(sentences))
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))
    if METRIC == 'cosine':
        similarity = similarity_matrix(sentences)
    # add edges to the graph (weighted by Levenshtein distance)
    for pair in nodePairs:
        firstInd = pair[0]
        secondInd = pair[1]
        if METRIC == 'cosine':
            weight = similarity[firstInd, secondInd]
        else:
            weight = METRIC(sentences[firstInd], sentences[secondInd])
        gr.add_edge(firstInd, secondInd, weight=weight)
    return gr

def textrank(abstracts):
    predictions, failed = [], []
    k = 0
    for abstract in abstracts:
        try:
            sentences = nltk.sent_tokenize(abstract)
            graph = build_graph(sentences)
            calculated_page_rank = nx.pagerank(graph, weight='weight')
            sentencesRank = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)
            predictions.append(sentences[sentencesRank[0]])
        except:
            predictions.append("")
            failed.append(k)
        print(k)
        k += 1
    return np.asarray(predictions), failed

SET = 2
MODEL = 3

metrics = ['cosine', levenshtein_distance, overlap, lcs]
METRIC = metrics[MODEL]
abstracts, titles = data.get_data()
titles_peer, failed = textrank(abstracts[SET])
titles_model = titles[SET]
scores = rouge.eval_rouge(titles_peer, titles_model)
print(SET, MODEL)
print(scores)
print(failed)