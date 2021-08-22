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
from ema import *

# Ashim
#from luka_loss import *
#from luka_loss_relu import *
from luka_loss_constrained import *
from godel_loss import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/chunking")
parser.add_argument('--train_data', help="Path to training data hdf5 file.", default="chunking-train.hdf5")
parser.add_argument('--val_data', help="Path to validation data hdf5 file.", default="")
parser.add_argument('--save_file', help="Path to where model to be saved.", default="model")
parser.add_argument('--word_vecs', help="Path to word vector hdf5 file", default="glove.hdf5")
parser.add_argument('--dict', help="The path to word dictionary", default = "chunking.word.dict")
parser.add_argument('--char_idx', help="The path to word2char index file", default = "char.idx.hdf5")
parser.add_argument('--char_dict', help="The path to char dictionary", default = "char.dict.txt")
parser.add_argument('--label_dict', help="Path to word dict file", default="chunking.label.dict")
# data
parser.add_argument('--percent', help="The percent of training data to use", type=float, default=1.0)
parser.add_argument('--div_percent', help="The percent of training data to divide as train/val", type=float, default=0.9)
parser.add_argument('--train_res', help="Path to training resource file.", default="")
parser.add_argument('--val_res', help="Path to validating resource file.", default="")
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
parser.add_argument('--learning_rate', help="The learning rate for training", type=float, default=0.0001)
parser.add_argument('--clip_epoch', help="The starting epoch to enable clip", type=int, default=1)
parser.add_argument('--clip', help="The norm2 threshold to clip, set it to negative to disable", type=float, default=-1.0)
parser.add_argument('--ema', help="Whether to use EMA", type=int, default=1)
parser.add_argument('--mu', help="The mu ratio used in EMA", type=float, default=0.999)
parser.add_argument('--adam_betas', help="The betas used in adam", default='0.9,0.999')
parser.add_argument('--epochs', help="The number of epoches for training", type=int, default=100)
parser.add_argument('--optim', help="The name of optimizer to use for training", default='adam')
parser.add_argument('--acc_batch_size', help="The accumulative batch size, -1 to disable", type=int, default=-1)
# constraints
parser.add_argument('--constr', help="The list of constraints", default="")
parser.add_argument('--rhos', help="The list of rhos for constraints", default='')
# TODO, param_init of uniform dist or normal dist???
parser.add_argument('--print_every', help="Print stats after this many batches", type=int, default=50)
parser.add_argument('--seed', help="The random seed", type=int, default=3435)
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)

# Ashim
parser.add_argument('--use_luka', help="For Lukasiewicz Loss", default=False, action='store_true')
parser.add_argument('--use_godel', help="For Godel Loss", default=False, action='store_true')
parser.add_argument('--constraints_lambda', help="Lambda for constraints", type=float, default=0.0)


# train batch by batch, accumulate batches until the size reaches acc_batch_size
def train_epoch(opt, shared, m, optim, ema, data, epoch_id, sub_idx):
    train_loss = 0.0
    num_ex = 0
    start_time = time.time()
    min_grad_norm2 = 1000000000000.0
    max_grad_norm2 = 0.0

    loss = None
    if opt.loss == 'loss':
        loss = Loss(opt, shared)
    elif opt.loss == 'crf':
        loss = CRFLoss(opt, shared)
    elif opt.loss == 'luka': # Ashim : For Lukasiewicz Loss
        print('Using Lukasiewicz Loss')
        loss = LukaLoss(opt, shared)
    elif opt.loss == 'godel':
        print('Using Godel Loss')
        loss = GodelLoss(opt, shared)

    #if opt.loss == 'godel' and epoch_id < 100:
    #    print('Using R-Prod')
    #    loss = Loss(opt, shared)


    data_size = len(sub_idx)
    batch_order = torch.randperm(data_size)
    batch_order = [sub_idx[idx] for idx in batch_order]

    acc_batch_size = 0
    m.train(True)
    loss.train(True)
    loss.begin_pass()
    m.begin_pass()
    for i in range(data_size):
        (data_name, source, char_source, batch_ex_idx, batch_l, source_l, label, res_map) = data[batch_order[i]]

        #print('Batch_l ' + str(batch_l))
        #print('Source_l ' +str(source_l))
        #print(source)

        wv_idx = Variable(source, requires_grad=False)
        cv_idx = Variable(char_source, requires_grad=False)
        y_gold = Variable(label, requires_grad=False)

        # update network parameters
        shared.epoch = epoch_id
        m.update_context(batch_ex_idx, batch_l, source_l, res_map)

        # forward pass
        output = m.forward(wv_idx, cv_idx)

        # loss
        if opt.loss == 'luka':
            batch_loss = loss(output, y_gold, constraints_lambda = opt.constraints_lambda)
        else:
            batch_loss = loss(output, y_gold, constraints_lambda = opt.constraints_lambda)

        # stats
        train_loss += float(batch_loss.data)
        num_ex += batch_l
        time_taken = time.time() - start_time
        acc_batch_size += batch_l

        # accumulate grads
        batch_loss.backward()

        # accumulate current batch until the rolled up batch size exceeds threshold or meet certain boundary
        if i == data_size-1 or acc_batch_size >= opt.acc_batch_size or (i+1) % opt.print_every == 0:
            grad_norm2 = optim.step(m, acc_batch_size)
            if opt.ema == 1:
                ema.step(m)

            # clear up grad
            m.zero_grad()
            acc_batch_size = 0

            # stats
            grad_norm2_avg = grad_norm2
            min_grad_norm2 = min(min_grad_norm2, grad_norm2_avg)
            max_grad_norm2 = max(max_grad_norm2, grad_norm2_avg)
            time_taken = time.time() - start_time
            loss_stats = loss.print_cur_stats()

            if (i+1) % opt.print_every == 0:
                stats = '{0}, Batch {1:.1f}k '.format(epoch_id+1, float(i+1)/1000)
                stats += 'Grad {0:.1f}/{1:.1f} '.format(min_grad_norm2, max_grad_norm2)
                stats += 'Loss {0:.4f} '.format(train_loss / num_ex)
                stats += loss.print_cur_stats()
                stats += 'Time {0:.1f}'.format(time_taken)
                print(stats)

    perf, extra_perf = loss.get_epoch_metric()

    m.end_pass()
    loss.end_pass()

    return perf, extra_perf, train_loss / num_ex, num_ex


def train(opt, shared, m, optim, ema, train_data, val_data):
    best_val_perf = 0.0
    test_perf = 0.0
    train_perfs = []
    val_perfs = []
    extra_perfs = []

    print('{0} batches in train set'.format(train_data.size()))
    if val_data is not None:
        print('{0} batches in dev set'.format(val_data.size()))
    else:
        print('no dev set specified, will split train set into train/dev folds')

    print('subsampling train set by {0}'.format(opt.percent))
    train_idx, train_num_ex = train_data.subsample(opt.percent, random=True)
    print('for the record, first 10 batches: {0}'.format(train_idx[:10]))

    val_idx = None
    val_num_ex = 0
    if val_data is None:
        val_data = train_data
        print('splitting train set into train/dev folds by {0}'.format(opt.div_percent))
        train_idx, val_idx, train_num_ex, val_num_ex = train_data.split(train_idx, opt.div_percent)
    else:
        val_idx, val_num_ex = val_data.subsample(1.0, random=False) # use all val data as dev set

    print('final train set has {0} batches {1} examples'.format(len(train_idx), train_num_ex))
    print('for the record, first 10 batches: {0}'.format(train_idx[:10]))
    print('final val set has {0} batches {1} examples'.format(len(val_idx), val_num_ex))
    print('for the record, first 10 batches: {0}'.format(val_idx[:10]))

    start = 0
    for i in range(start, opt.epochs):
        train_perf, extra_train_perf, loss, num_ex = train_epoch(opt, shared, m, optim, ema, train_data, i, train_idx)
        train_perfs.append(train_perf)
        extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_train_perf])
        print('Train {0:.4f} All {1}'.format(train_perf, extra_perf_str))

        # evaluate
        #   and save if it's the best model
        val_perf, extra_val_perf, val_loss, num_ex = validate(opt, shared, m, val_data, val_idx)
        val_perfs.append(val_perf)
        extra_perfs.append(extra_val_perf)
        extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_val_perf])
        print('Val {0:.4f} All {1}'.format(val_perf, extra_perf_str))

        perf_table_str = ''
        cnt = 0
        print('Epoch  | Train | Val ...')
        for train_perf, extra_perf in zip(train_perfs, extra_perfs):
            extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
            perf_table_str += '{0}\t{1:.4f}\t{2}\n'.format(cnt+1, train_perf, extra_perf_str)
            cnt += 1
        print(perf_table_str)

        if val_perf > best_val_perf:
            best_val_perf = val_perf
            print('saving model to {0}'.format(opt.save_file))
            param_dict = m.get_param_dict()
            save_param_dict(param_dict, '{0}.hdf5'.format(opt.save_file))
            save_opt(opt, '{0}.opt'.format(opt.save_file))
            # save ema
            if opt.ema == 1:
                ema_param_dict = ema.get_param_dict()
                save_param_dict(ema_param_dict, '{0}.ema.hdf5'.format(opt.save_file))

        else:
            print('skip saving model for perf <= {0:.4f}'.format(best_val_perf))

def validate(opt, shared, m, val_data, val_idx):
    m.train(False)

    val_loss = 0.0
    num_ex = 0

    loss = None
    if opt.loss == 'loss':
        loss = Loss(opt, shared)
    elif opt.loss == 'crf':
        loss = CRFLoss(opt, shared)
    elif opt.loss == 'luka': # Ashim : For Lukasiewicz Loss
        print('Using Lukasiewicz Loss')
        loss = LukaLoss(opt, shared)
    elif opt.loss == 'godel':
        print('Using Godel Loss')
        loss = GodelLoss(opt, shared)


    loss.train(False)

    print('validating on the {0} batches...'.format(len(val_idx)))

    m.begin_pass()
    for i in range(len(val_idx)):
        (data_name, source, char_source, batch_ex_idx, batch_l, source_l, label, res_map) = val_data[val_idx[i]]

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
    m.end_pass()

    return (perf, extra_perf, val_loss / num_ex, num_ex)




def main(args):
    opt = parser.parse_args(args)
    shared = Holder()

    #
    opt.train_data = opt.dir + opt.train_data
    opt.val_data = opt.dir + opt.val_data
    opt.train_res = '' if opt.train_res == ''  else ','.join([opt.dir + path for path in opt.train_res.split(',')])
    opt.val_res = '' if opt.val_res == ''  else ','.join([opt.dir + path for path in opt.val_res.split(',')])
    opt.word_vecs = opt.dir + opt.word_vecs
    opt.char_idx = opt.dir + opt.char_idx
    opt.dict = opt.dir + opt.dict
    opt.char_dict = opt.dir + opt.char_dict
    opt.label_dict = opt.dir + opt.label_dict

    torch.manual_seed(opt.seed)
    if opt.gpuid != -1:
        torch.cuda.set_device(opt.gpuid)
        torch.cuda.manual_seed_all(opt.seed)

    print(opt)

    # build model
    m = Pipeline(opt, shared)
    optim = Optimizer(opt, shared)
    ema = EMA(opt, shared)

    m.init_weight()
    # Initialize weights
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print('total number of trainable parameters: {0}'.format(num_params))
    if opt.gpuid != -1:
        m = m.cuda()

    # loading data
    train_data = Data(opt, opt.train_data, None if opt.train_res == '' else opt.train_res.split(','))

    val_data = None
    if opt.val_data != opt.dir:
        val_data = Data(opt, opt.val_data, None if opt.val_res == '' else opt.val_res.split(','))

    train(opt, shared, m, optim, ema, train_data, val_data)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
