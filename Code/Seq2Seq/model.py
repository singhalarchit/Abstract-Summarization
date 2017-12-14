# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:54:42 2017

@author: Archit

Code adapted from https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/model.py
"""

"""Sequence to Sequence models."""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Seq2SeqAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(self, args):
        """Initialize model."""
        super(Seq2SeqAttention, self).__init__()
        self.bidirectional = args.bidirectional
        self.src_hidden_dim = args.src_hidden_dim
        self.src_hidden_dim = self.src_hidden_dim // 2 \
            if self.bidirectional else self.src_hidden_dim
        self.num_directions = 2 if self.bidirectional else 1
        self.src_embedding = nn.Embedding(args.src_vocab_size, args.src_emb_dim)
        self.trg_embedding = nn.Embedding(args.trg_vocab_size, args.trg_emb_dim)
        self.encoder = nn.LSTM(args.src_emb_dim, 
                               self.src_hidden_dim, 
                               args.src_num_layers,
                               bidirectional = self.bidirectional,
                               dropout = args.dropout)
        self.decoder = LSTMAttentionDot(args.trg_emb_dim, args.trg_hidden_dim)
        self.encoder2decoder = nn.Linear(
                self.src_hidden_dim * self.num_directions, args.trg_hidden_dim)
        self.decoder2vocab = nn.Linear(args.trg_hidden_dim, args.trg_vocab_size)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)
        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)

        return h0_encoder.cuda(), c0_encoder.cuda()

    def forward(self, input_src, input_trg):
        """Propogate input through the network."""
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)

        self.h0_encoder, self.c0_encoder = self.get_state(input_src)

        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (self.h0_encoder, self.c0_encoder)
        )

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]
        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        ctx = src_h.transpose(0, 1)

        trg_h, (_, _) = self.decoder(
            trg_emb,
            (decoder_init_state, c_t),
            ctx
        )

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size()[0] * trg_h.size()[1],
            trg_h.size()[2]
        )
        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
        )
        return decoder_logit

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.trg_vocab_size)
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs

    
class LSTMAttentionDot(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size):
        """Initialize params."""
        super(LSTMAttentionDot, self).__init__()
        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)

    def forward(self, input, hidden, ctx):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            h_tilde, alpha = self.attention_layer(hy, ctx.transpose(0, 1))

            return h_tilde, cy

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return output, hidden


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn