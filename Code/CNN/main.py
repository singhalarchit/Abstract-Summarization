# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:24:39 2017

@author: Archit
"""

#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import model
import train
#import sys
#sys.path.append('../')
from data_summary import load_vocab as load_vocab
from load_data import get_splitted_data as get_splitted_data

parser = argparse.ArgumentParser(description='Abstract Summarizer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=10000, help='number of epochs for train [default: 256]')
parser.add_argument('-log-interval',  type=int, default=100,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-plot-interval',  type=int, default=50,   help='how many steps to wait before plotting training status [default: 1]')
#parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=1000, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='../Snapshots', help='where to save the snapshots')
# data 
#parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
# model
parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.1]')
#parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='2,3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-hidden-dim', type=int, default=128, help='number of hidden dimension [default: 256]')
parser.add_argument('-num-layers', type=int, default=1, help='number of hidden layers in RNN [default: 1]')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
#parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
#parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
#parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


# Load Vocab
word2idx, _, idx2titleidx, _ = load_vocab()


# update args and print
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
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


"""
# model
if args.snapshot is None:
    encoderCNN = model.EncoderCNN(args)
    encoderRNN = model.EncoderRNN(args)
    decoder = model.AttnDecoderRNN(args, encoderCNN)
else:
    print('\nLoading models from [%s]...' % args.snapshot)
    try:
        encoderCNN = torch.load(args.snapshot + 'encoderCNN.pt')
        encoderRNN = torch.load(args.snapshot + 'encoderRNN.pt')
        decoder = torch.load(args.snapshot + 'decoder.pt')
    except:
        print("Sorry, The snapshot(s) doesn't exist."); exit()

if args.cuda:
    encoderCNN = encoderCNN.cuda()
    encoderRNN = encoderRNN.cuda()
    decoder = decoder.cuda()
"""

encoderCNN = model.EncoderCNN(args)
encoderRNN = model.EncoderRNN(args)
decoder = model.AttnDecoderRNN(args, encoderCNN)
train.trainIters1(args, abstracts[0][:100], titles[0][:100], encoderCNN,
                 encoderRNN, decoder)

"""
vanillaEncoderRNN = model.VanillaEncoderRNN(args)
vanillaDecoderRNN = model.VanillaDecoderRNN(args, vanillaEncoderRNN)
train.trainIters(args, abstracts[0][:100], titles[0][:100], vanillaEncoderRNN,
                 vanillaDecoderRNN)
"""

"""
filename = "../Snapshots/2017-12-01_02-00-03/"
step = '1200'
"""

"""
vanillaEncoderRNN = torch.load(filename + 'vanillaEncoderRNN_steps' + step + '.pt')
vanillaDecoderRNN = torch.load(filename + 'vanillaDecoderRNN_steps' + step + '.pt')
train.evaluateRandomly(args, abstracts[0][:100], titles[0][:100],
                       vanillaEncoderRNN, vanillaDecoderRNN, n = 3)
"""

"""
encoderCNN = torch.load(filename + 'encoderCNN_steps' + step + '.pt')
encoderRNN = torch.load(filename + 'encoderRNN_steps' + step + '.pt')
decoder = torch.load(filename + 'decoder_steps' + step + '.pt')
"""
train.evaluateRandomly1(args, abstracts[0][:100], titles[0][:100], encoderCNN, 
                       encoderRNN, decoder, n = 3)


'''
# train or predict
if args.predict is not None:
    label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_iter, dev_iter, cnn, args)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
'''
