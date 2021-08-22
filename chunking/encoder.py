import sys

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *


class Encoder(torch.nn.Module):
    def __init__(self, opt, shared):
        super(Encoder, self).__init__()
        self.opt = opt
        self.shared = shared

        rnn_input_size = opt.word_vec_size
        rnn_hidden_size = opt.hidden_size if opt.bidir == 0 else opt.hidden_size//2
        self.rnn = build_rnn(
            opt.rnn_type,
            input_size = rnn_input_size,
            hidden_size = rnn_hidden_size,
            num_layers = 1,
            bias = True,
            batch_first = True,
            dropout = 0.0,
            bidirectional = opt.bidir==1)
        self.drop = nn.Dropout(opt.dropout) # TODO, locked dropout probably be better


    def forward(self, sent, char_sent):
        sent, _ = self.rnn(sent)
        sent = self.drop(sent)
        return sent

    def begin_pass(self):
        pass

    def end_pass(self):
        pass
