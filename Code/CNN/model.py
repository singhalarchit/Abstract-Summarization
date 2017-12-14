# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:25:55 2017

@author: Archit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)

class  EncoderCNN(nn.Module):
    def __init__(self, args, embedding):
        super(EncoderCNN, self).__init__()
        self.args = args
        D = args.embed_dim                  # embed_dim for a word
        self.Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        #D2 = Co * len(Ks)                  # embed_dim for a sentence
        self.embed = embedding
        self.convs1 = nn.ModuleList([nn.Conv2d(self.Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(self.args.dropout)
        #self.convs2 = nn.ModuleList([nn.Conv2d(self.Ci, Co, (K, D2)) for K in Ks]) #padding=(K-1, 0)

    def feature_maps(self, x, convs):
        num_words = x.size(2)
        temp = Variable(torch.zeros((self.Ci, self.args.kernel_num, 1)))
        temp = temp.cuda() if self.args.cuda else temp
        y = []
        for conv, k in zip(convs, self.args.kernel_sizes):
            if k > num_words:
                y.append(temp)
            else:
                y.append(F.relu(conv(x)).squeeze(3))
        return y

    def sent_enc(self, x):
        # x is a list of word indexes
        x = Variable(torch.LongTensor([x]))
        x = x.cuda() if self.args.cuda else x
        x = self.embed(x) # (N,W,D) (1,#words,D)
        if self.args.static:
            x = Variable(x.data)
        x = x.unsqueeze(1) # (N,Ci,W,D) (1,1,#words,D)
        x = self.feature_maps(x, self.convs1)
        #x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] # [(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1) # (N,len(Ks)*Co)
        return x

    """
    def abstract_enc(self, abstract):
        y = [self.sent_enc(x) for x in abstract]  # [(1,len(Ks)*Co), ...]*#sentences
        y = torch.cat(y, 0)  # (#sentences,len(Ks)*Co)
        y = y.unsqueeze(0).unsqueeze(0) # (1,Ci,#sentences,D2)
        y = self.feature_maps(y, self.convs2)
        #y = [F.relu(conv(y)).squeeze(3) for conv in self.convs2] # [(1,Co,#sentences), ...]*len(Ks)
        y = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y] # [(1,Co), ...]*len(Ks)
        y = torch.cat(y, 1) # (1,len(Ks)*Co)
        return y
    """


class EncoderRNN(nn.Module):
    def __init__(self, args):
        super(EncoderRNN, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_dim
        input_size = args.kernel_num * len(args.kernel_sizes)
        self.gru = nn.GRU(input_size, self.hidden_size)

    def forward(self, input, hidden):
        output = input.view(1, 1, -1)
        for i in range(self.args.num_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.args.cuda:
            return result.cuda()
        else:
            return result


class VanillaEncoderRNN(nn.Module):
    def __init__(self, args, embedding):
        super(VanillaEncoderRNN, self).__init__()
        self.args = args
        self.embedding = embedding
        self.hidden_size = args.hidden_dim
        self.gru = nn.GRU(args.embed_dim, self.hidden_size)

    def forward(self, input, hidden):
        input = Variable(torch.LongTensor([[input]]))
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.args.num_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.args.cuda:
            return result.cuda()
        else:
            return result


class VanillaDecoderRNN(nn.Module):
    def __init__(self, args, embedding):
        super(VanillaDecoderRNN, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_dim
        self.embedding = embedding
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.args.output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.args.num_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.args.cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN2(nn.Module):
    def __init__(self, args, embedding):
        super(AttnDecoderRNN2, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_dim
        self.embedding = embedding
        self.dropout = nn.Dropout(self.args.dropout)
        self.attn = nn.Linear(self.hidden_size * 2, self.args.max_sentences)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.args.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))  # 1 x max_sentences
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))  # 1 x 1 x hidden_size
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)         # 1 x 1 x hidden_size
        for i in range(self.args.num_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights
    
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.args.cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, args, embedding):
        super(AttnDecoderRNN, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_dim
        self.embedding = embedding
        self.dropout = nn.Dropout(self.args.dropout)
        self.gru = nn.GRU(args.embed_dim, self.hidden_size)
        p, q = 100, 200
        self.z = nn.Parameter(torch.randn(1, p))
        self.v = nn.Parameter(torch.randn(1, q))
        self.We = nn.Parameter(torch.randn(self.hidden_size, p))
        self.Wr = nn.Parameter(torch.randn(self.hidden_size, p))
        self.We2 = nn.Parameter(torch.randn(self.hidden_size, q))
        self.Wr2 = nn.Parameter(torch.randn(args.embed_dim, q))

    def forward(self, input, hidden, encoder_outputs, word_embeddings):
        embedded = self.embedding(input).view(1, 1, -1)
        output = self.dropout(embedded)
        for i in range(self.args.num_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        a = F.tanh(torch.mm(hidden.squeeze(0), self.We) + 
                   torch.mm(encoder_outputs, self.Wr))
        b = F.softmax(torch.mm(self.z, torch.t(a)))
        encoder_output = torch.mm(b, encoder_outputs)
        u = F.tanh(torch.mm(encoder_output, self.We2) + 
                   torch.mm(word_embeddings, self.Wr2))
        scores = torch.mm(self.v, torch.t(u))
        p = F.log_softmax(scores)        
        return output, hidden, b, p
    
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.args.cuda:
            return result.cuda()
        else:
            return result
