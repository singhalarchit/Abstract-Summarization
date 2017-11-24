# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:56:24 2017

@author: Archit
"""

import numpy as np

def read_dataset(filename):
    dataset = []
    with open(filename) as f:
        for line in f:
            line = line.decode('utf-8').encode('ascii','ignore')
            nodigit = ''.join([i for i in line if not i.isdigit()])
            dataset.append(nodigit)
    return dataset

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

def get_data():
    abstract_file = '../Processed Data/abstracts.txt'
    title_file = '../Processed Data/titles.txt'
    abstracts, titles = read_dataset(abstract_file), read_dataset(title_file)
    abstracts_split, titles_split = split_data(abstracts, titles)
    #15874, 1984, 1985
    return abstracts_split[:3], titles_split[:3]

#failure examples: 13018 (W07-1426), 13441 (W08-2205), 15250 (W11-2102)
#abstracts, titles = get_data()