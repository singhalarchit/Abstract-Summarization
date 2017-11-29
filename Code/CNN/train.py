# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 23:07:29 2017

@author: Archit
"""

import datetime
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import os
import random
import time
import torch
import torch.nn as nn
import warnings
import sys
sys.path.append('../')
from data_summary import load_vocab as load_vocab
from torch.autograd import Variable
warnings.filterwarnings("ignore")


word2idx, idx2word, idx2titleidx, titleidx2idx = load_vocab()
SOS_token, EOS_token = word2idx['<SOS>'], word2idx['<EOS>']
stopwords = ['for','of','and','a','in','the','to']
stopwordsidx = [word2idx[word] for word in stopwords]


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('NLL Loss', fontsize=12)
    plt.title('NLL Loss vs Number of iterations')
    plt.grid()
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig('../Plots/training_curve_' + date, bbox_inches='tight')
    plt.close()
    
def variable(x):
    return Variable(torch.LongTensor([x]))


def train2(args, abstract, title, encoderCNN, encoderRNN, decoder, 
          encoderCNN_optimizer, encoderRNN_optimizer, decoder_optimizer, 
          criterion):
    #print('abstract: ', abstract)
    #print('title: ', title)
    title = [idx for idx in title if not idx in stopwordsidx]
    encoderRNN_hidden = encoderRNN.initHidden()
    encoderCNN_optimizer.zero_grad()
    encoderRNN_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Number of sentences
    num_sentences = len(abstract)
    # Number of words in gold title (capped to max_length)
    target_length = args.max_length if len(title) > args.max_length else len(title)
    title.append(EOS_token)
    target_length += 1
    #print('num_sentences: ', num_sentences, 'target_length: ', target_length)
    encoderRNN_outputs = Variable(
            torch.zeros(args.max_sentences, encoderRNN.hidden_size))
    encoderRNN_outputs = encoderRNN_outputs.cuda() if args.cuda else encoderRNN_outputs
    loss = 0
    for ei in range(num_sentences):
        encoderRNN_input = encoderCNN.sent_enc(abstract[ei])
        encoderRNN_input = encoderCNN.dropout(encoderRNN_input)
        encoderRNN_output, encoderRNN_hidden = encoderRNN(
                encoderRNN_input, encoderRNN_hidden)
        encoderRNN_outputs[ei] = encoderRNN_output[0][0]
    decoder_input = variable([SOS_token])
    decoder_input = decoder_input.cuda() if args.cuda else decoder_input
    decoder_hidden = encoderRNN_hidden
    use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False    
    #print('use_teacher_forcing: ', use_teacher_forcing)
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoderRNN_outputs)
            loss += criterion(decoder_output, variable(idx2titleidx[title[di]]))
            decoder_input = variable([title[di]])  # Teacher forcing
            decoder_input = decoder_input.cuda() if args.cuda else decoder_input
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoderRNN_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = titleidx2idx[topi[0][0]]
            decoder_input = variable([ni])
            decoder_input = decoder_input.cuda() if args.cuda else decoder_input
            loss += criterion(decoder_output, variable(idx2titleidx[title[di]]))
            if ni == EOS_token:
                print("EOS encounted")
                break
    loss.backward()
    encoderCNN_optimizer.step()
    encoderRNN_optimizer.step()
    decoder_optimizer.step()
    return loss.data[0] / target_length


def train(args, abstract, title, encoderCNN, encoderRNN, decoder, 
          encoderCNN_optimizer, encoderRNN_optimizer, decoder_optimizer, 
          criterion):
    encoderRNN_hidden = encoderRNN.initHidden()
    encoderCNN_optimizer.zero_grad()
    encoderRNN_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    abstract_idx = [idx for sentence in abstract for idx in sentence]
    abstract_idxs = Variable(torch.LongTensor(abstract_idx))
    abstract_idxs = abstract_idxs.cuda() if args.cuda else abstract_idxs
    word_embeddings = encoderCNN.embed(abstract_idxs)
    num_sentences = len(abstract)
    target_length = args.max_length if len(title) > args.max_length else len(title)
    title.append(EOS_token)
    target_length += 1
    abstract_idx.extend(title)
    encoderRNN_outputs = Variable(
            torch.zeros(args.max_sentences, encoderRNN.hidden_size))
    encoderRNN_outputs = encoderRNN_outputs.cuda() if args.cuda else encoderRNN_outputs
    loss = 0
    for ei in range(num_sentences):
        encoderRNN_input = encoderCNN.sent_enc(abstract[ei])
        encoderRNN_input = encoderCNN.dropout(encoderRNN_input)
        encoderRNN_output, encoderRNN_hidden = encoderRNN(
                encoderRNN_input, encoderRNN_hidden)
        encoderRNN_outputs[ei] = encoderRNN_output[0][0]
    decoder_input = variable([SOS_token])
    decoder_input = decoder_input.cuda() if args.cuda else decoder_input
    decoder_hidden = encoderRNN_hidden
    use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False    
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            _, decoder_hidden, _, p = decoder(decoder_input, decoder_hidden, 
                                           encoderRNN_outputs, word_embeddings)
            #dummy_p = Variable(torch.zeros((1, target_length)))
            #output_prob = torch.cat((p, dummy_p), 1)
            true_label = abstract_idx.index(title[di])
            loss += torch.max(-p) if true_label > (p.size(1) - 1) else -p[0][true_label]            
            #loss += criterion(p, variable(true_label))
            decoder_input = variable([title[di]])  # Teacher forcing
            decoder_input = decoder_input.cuda() if args.cuda else decoder_input
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            _, decoder_hidden, _, p = decoder(decoder_input, decoder_hidden, 
                                           encoderRNN_outputs, word_embeddings)
            #dummy_p = Variable(torch.zeros((1, target_length)))
            #output_prob = torch.cat((p, dummy_p), 1)
            true_label = abstract_idx.index(title[di])
            loss += torch.max(-p) if true_label > (p.size(1) - 1) else -p[0][true_label]
            topv, topi = p.data.topk(1)
            ni = abstract_idx[topi[0][0]]
            decoder_input = variable([ni])
            decoder_input = decoder_input.cuda() if args.cuda else decoder_input
            #loss += criterion(p, variable(true_label))
            if ni == EOS_token:
                print("EOS encounted at ", di)
                break
    loss.backward()
    encoderCNN_optimizer.step()
    encoderRNN_optimizer.step()
    decoder_optimizer.step()
    return loss.data[0] / target_length


def trainIters(args, abstracts, titles, encoderCNN, encoderRNN, decoder):
    start = time.time()
    plot_losses = []
    print_loss_total = 0    # Reset every args.log_interval
    plot_loss_total = 0     # Reset every args.plot_interval
    encoderCNN_optimizer = torch.optim.SGD(encoderCNN.parameters(), lr = args.lr)
    encoderRNN_optimizer = torch.optim.SGD(encoderRNN.parameters(), lr = args.lr)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr = args.lr)
    criterion = nn.NLLLoss()
    for iter in range(1, args.epochs + 1):
        abstract, title = abstracts[iter % len(titles) - 1], titles[iter % len(titles) - 1]
        loss = train(args, abstract, title, encoderCNN, encoderRNN, decoder, 
                     encoderCNN_optimizer, encoderRNN_optimizer,
                     decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        if iter % args.log_interval == 0:
            log = (torch.sum((encoderCNN.embed.weight == decoder.embedding.weight).int())/40648).data[0]
            print_loss_avg = print_loss_total / args.log_interval
            print_loss_total = 0
            print('%s (%d %d%%) %.4f %d' % (timeSince(start, float(iter) / args.epochs),
                       iter, float(iter) / args.epochs * 100, print_loss_avg, log))
        if iter % args.plot_interval == 0:
            plot_loss_avg = plot_loss_total / args.plot_interval
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        if iter % args.save_interval == 0:
            if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
            models = [encoderCNN, encoderRNN, decoder]
            prefixes = ['encoderCNN', 'encoderRNN', 'decoder']
            for model, prefix in zip(models, prefixes):
                save_prefix = os.path.join(args.save_dir, prefix)
                save_path = '{}_steps{}.pt'.format(save_prefix, iter)
                torch.save(model, save_path)
    showPlot(plot_losses)


def evaluate2(args, abstract, encoderCNN, encoderRNN, decoder):
    num_sentences = len(abstract)
    encoderRNN_hidden = encoderRNN.initHidden()
    encoderRNN_outputs = Variable(
            torch.zeros(args.max_sentences, encoderRNN.hidden_size))
    encoderRNN_outputs = encoderRNN_outputs.cuda() if args.cuda else encoderRNN_outputs
    for ei in range(num_sentences):
        encoderRNN_input = encoderCNN.sent_enc(abstract[ei])
        encoderRNN_input = encoderRNN_input * args.dropout
        encoderRNN_output, encoderRNN_hidden = encoderRNN(
                encoderRNN_input, encoderRNN_hidden)
        encoderRNN_outputs[ei] = encoderRNN_output[0][0]
    decoder_input = variable([SOS_token])  # SOS
    decoder_input = decoder_input.cuda() if args.cuda else decoder_input
    decoder_hidden = encoderRNN_hidden
    print(decoder_hidden.data[0][0][:10].numpy())
    decoded_words = []    
    decoder_attentions = torch.zeros(args.max_length, args.max_sentences)
    for di in range(args.max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoderRNN_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = titleidx2idx[topi[0][0]]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(idx2word[ni])
        decoder_input = variable([ni])
        decoder_input = decoder_input.cuda() if args.cuda else decoder_input
    return decoded_words, decoder_attentions[:di + 1]


def evaluate(args, abstract, encoderCNN, encoderRNN, decoder):
    abstract_idx = [idx for sentence in abstract for idx in sentence]
    abstract_idxs = Variable(torch.LongTensor(abstract_idx))
    abstract_idxs = abstract_idxs.cuda() if args.cuda else abstract_idxs
    word_embeddings = encoderCNN.embed(abstract_idxs)
    num_sentences = len(abstract)
    encoderRNN_hidden = encoderRNN.initHidden()
    encoderRNN_outputs = Variable(
            torch.zeros(args.max_sentences, encoderRNN.hidden_size))
    encoderRNN_outputs = encoderRNN_outputs.cuda() if args.cuda else encoderRNN_outputs
    for ei in range(num_sentences):
        encoderRNN_input = encoderCNN.sent_enc(abstract[ei])
        encoderRNN_input = encoderRNN_input * args.dropout
        encoderRNN_output, encoderRNN_hidden = encoderRNN(
                encoderRNN_input, encoderRNN_hidden)
        encoderRNN_outputs[ei] = encoderRNN_output[0][0]
    decoder_input = variable([SOS_token])  # SOS
    decoder_input = decoder_input.cuda() if args.cuda else decoder_input
    decoder_hidden = encoderRNN_hidden
    print(decoder_hidden.data[0][0][:10].numpy())
    decoded_words = []    
    decoder_attentions = torch.zeros(args.max_length, args.max_sentences)
    for di in range(args.max_length):
        _, decoder_hidden, b, p = decoder(decoder_input, decoder_hidden, 
                                       encoderRNN_outputs, word_embeddings)
        decoder_attentions[di] = b.data
        topv, topi = p.data.topk(1)
        ni = abstract_idx[topi[0][0]]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(idx2word[ni])
        decoder_input = variable([ni])
        decoder_input = decoder_input.cuda() if args.cuda else decoder_input
    return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(args, abstracts, titles, encoderCNN, encoderRNN, decoder, 
                     n = 10):
    total = len(titles)
    for i in range(n):
        ind = random.randint(0, total - 1)
        abstract, title = abstracts[ind], titles[ind]
        title = [idx2word[idx] for idx in title]
        title = ' '.join(title[:25])
        abstractt = [idx2word[idx] for sentence in abstract for idx in sentence]
        abstractt = ' '.join(abstractt[:25])
        print('>', abstractt)
        print('=', title)
        output_words, attentions = evaluate(args, abstract, encoderCNN,
                                            encoderRNN, decoder)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def attentionVisualize(args, abstract, encoderCNN, encoderRNN, decoder):
    output_words, attentions = evaluate(args, abstract, encoderCNN, encoderRNN, 
                                        decoder)
    plt.matshow(attentions.numpy())


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()