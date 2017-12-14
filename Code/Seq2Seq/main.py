# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:04:38 2017

@author: Archit
"""

#! /usr/bin/env python
import argparse
import datetime
import model
import os
import sys
import torch
import train
sys.path.append('../')
from data_summary import load_vocab as load_vocab
from load_data import get_splitted_data as get_splitted_data
from model import Seq2SeqAttention


parser = argparse.ArgumentParser()
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=10000, help='number of epochs for train [default: 256]')
parser.add_argument('-log-interval',  type=int, default=100,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-plot-interval',  type=int, default=50,   help='how many steps to wait before plotting training status [default: 1]')
#parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=100, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='../Snapshots', help='where to save the snapshots')
# data 
#parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
# model
parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.1]')
#parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-src-emb-dim', type=int, default=128, help='number of embedding dimension of source [default: 128]')
parser.add_argument('-trg-emb-dim', type=int, default=128, help='number of embedding dimension of target [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='2,3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-src-hidden-dim', type=int, default=128, help='number of hidden dimension in source RNN [default: 256]')
parser.add_argument('-src-num-layers', type=int, default=1, help='number of hidden layers in source RNN [default: 1]')
parser.add_argument('-trg-hidden-dim', type=int, default=128, help='number of hidden dimension in target RNN [default: 256]')
parser.add_argument('-trg-num-layers', type=int, default=1, help='number of hidden layers in target RNN [default: 1]')
parser.add_argument('-ctx-hidden-dim', type=int, default=128, help='number of hidden dimension for context [default: 256]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
#parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
#parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


# Load Vocab
word2idx, _, idx2titleidx, _ = load_vocab()


# update args and print
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
args.embed_num = len(word2idx)
args.output_size = len(idx2titleidx)
args.teacher_forcing_ratio = 0.5
args.max_sentences = 10         # Maximum number of sentences in each abstract
args.max_length = 25            # Maximum number of words in a title


print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# Load dataset
abstracts, titles = get_splitted_data()

args.src_vocab_size = 100
args.trg_vocab_size = 100
args.bidirectional = False



loss_criterion = nn.CrossEntropyLoss()

model = Seq2SeqAttention(args)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

decoder_logit = model(input_lines_src, input_lines_trg)
optimizer.zero_grad()

loss = loss_criterion(decoder_logit.contiguous().view(-1, vocab_size),
                      output_lines_trg.view(-1))
losses.append(loss.data[0])
loss.backward()
optimizer.step()
