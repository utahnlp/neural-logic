import sys
import argparse
import h5py
import os
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch import cuda
from holder import *
from pipeline import *
from loss import *
from crf_loss import *
from optimizer import *
from data import *
from util import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/chunking/")
parser.add_argument('--data', help="Path to training data hdf5 file.", default="chunking-test.hdf5")
parser.add_argument('--load_file', help="Path to where model to be saved.", default="")
parser.add_argument('--word_vecs', help="Path to word vector hdf5 file", default="glove.hdf5")
parser.add_argument('--dict', help="The path to word dictionary", default = "chunking.word.dict")
parser.add_argument('--char_idx', help="The path to word2char index file", default = "char.idx.hdf5")
parser.add_argument('--char_dict', help="The path to char dictionary", default = "char.dict.txt")
parser.add_argument('--label_dict', help="Path to word dict file", default="chunking.label.dict")
# data
parser.add_argument('--res', help="Path to training resource file.", default="")
## pipeline specs
parser.add_argument('--use_char_enc', help="Whether to use char encoding", type=int, default=0)
parser.add_argument('--char_emb_size', help="The size of char embedding", type=int, default=20)
parser.add_argument('--char_enc_size', help="The size of char encoding", type=int, default=100)
parser.add_argument('--char_encoder', help="The type of char encoder, cnn/rnn", default='cnn')
parser.add_argument('--char_filters', help="The list of filters for char cnn", default='5')
parser.add_argument('--num_char', help="The number of distinct chars", type=int, default=58)
parser.add_argument('--hw_layer', help="The number of highway layers to use", type=int, default=2)

parser.add_argument('--rnn_type', help="The type of rnn to use (lstm or gru)", default='lstm')
parser.add_argument('--word_vec_size', help="The input word embedding dim", type=int, default=300)
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=200)
parser.add_argument('--token_l', help="The maximal token length", type=int, default=16)
parser.add_argument('--bidir', help="Whether to use bidirectional rnn", type=int, default=1)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.5)
parser.add_argument('--num_label', help="The number of fine BIO labels", type=int, default=23)
parser.add_argument('--use_crf', help="Whether to use crf", type=int, default=0)
parser.add_argument('--loss', help="The type of loss function", default='loss')
parser.add_argument('--use_f1', help="Whether to use F1 as metrix", type=int, default=0)
# learning
parser.add_argument('--fix_word_vecs', help="Whether to make word embeddings NOT learnable", type=int, default=1)
parser.add_argument('--param_init_type', help="The type of initializer", default='xavier_uniform')
# constraints
parser.add_argument('--constr', help="The list of constraints", default="")
parser.add_argument('--rhos', help="The list of rhos for constraints", default='')
# TODO, param_init of uniform dist or normal dist???
parser.add_argument('--print_every', help="Print stats after this many batches", type=int, default=50)
parser.add_argument('--seed', help="The random seed", type=int, default=3435)
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--verbose', help="Whether to print out some log", type=int, default=0)
parser.add_argument('--print', help="Prefix to where verbose printing will be piped", default='print')

parser.add_argument('--use_luka', help="For Lukasiewicz Loss", default=False, action='store_true')
parser.add_argument('--use_godel', help="For Godel Loss", default=False, action='store_true')

def evaluate(opt, shared, m, data):
    m.train(False)

    val_loss = 0.0
    num_ex = 0
    verbose = opt.verbose==1

    loss = None
    if opt.loss == 'loss':
        loss = Loss(opt, shared)
    elif opt.loss == 'crf':
        loss = CRFLoss(opt, shared)
    loss.train(False)

    loss.verbose = verbose

    m.begin_pass()
    loss.begin_pass()
    for i in range(data.size()):
        (data_name, source, char_source, batch_ex_idx, batch_l, source_l, label, res_map) = data[i]

        wv_idx = Variable(source, requires_grad=False)
        cv_idx = Variable(char_source, requires_grad=False)
        y_gold = Variable(label, requires_grad=False)

        # update network parameters
        m.update_context(batch_ex_idx, batch_l, source_l, res_map)

        # forward pass
        pred = m.forward(wv_idx, cv_idx)

        # loss
        batch_loss = loss(pred, y_gold)

        # stats
        val_loss += float(batch_loss.data)
        num_ex += batch_l

    perf, extra_perf = loss.get_epoch_metric()
    loss.end_pass()
    m.end_pass()

    return (perf, extra_perf, val_loss / num_ex, num_ex)



def main(args):
    opt = parser.parse_args(args)
    shared = Holder()

    #
    opt.data = opt.dir + opt.data
    opt.res = '' if opt.res == ''  else ','.join([opt.dir + path for path in opt.res.split(',')])
    opt.word_vecs = opt.dir + opt.word_vecs
    opt.char_idx = opt.dir + opt.char_idx
    opt.dict = opt.dir + opt.dict
    opt.char_dict = opt.dir + opt.char_dict
    opt.label_dict = opt.dir + opt.label_dict

    if opt.gpuid != -1:
        torch.cuda.set_device(opt.gpuid)
        torch.cuda.manual_seed_all(1)

    # build model
    m = Pipeline(opt, shared)

    # initialization
    print('loading pretrained model from {0}...'.format(opt.load_file))
    param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
    m.set_param_dict(param_dict)

    if opt.gpuid != -1:
        m = m.cuda()

    # loading data
    res_files = None if opt.res == '' else opt.res.split(',')
    data = Data(opt, opt.data, res_files)

    #
    perf, extra_perf, avg_loss, num_ex = evaluate(opt, shared, m, data)
    extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
    print('Val {0:.4f} Extra {1} Loss: {2:.4f}'.format(
        perf, extra_perf_str, avg_loss))

    #print('saving model to {0}'.format('tmp'))
    #param_dict = m.get_param_dict()
    #save_param_dict(param_dict, '{0}.hdf5'.format('tmp'))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
