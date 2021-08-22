import sys

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
import numpy as np
from official_eval import *


# Lukasiewicz Loss
class LukaLoss(torch.nn.Module):
    def __init__(self, opt, shared):
        super(LukaLoss, self).__init__()
        self.opt = opt
        self.shared = shared
        # do not create loss node globally
        self.num_correct = 0
        self.num_all = 0
        self.num_ex = 0
        self.verbose = False
        self.all_pred = []
        self.all_pos_pred = []

        # for official eval of f1 in NER
        self.all_pred_label = []
        self.all_gold_label = []
        self.idx_to_label = self.load_label_map()

        # load dict entries
        self.label_str = []
        with open(self.opt.label_dict, 'r') as f:
            for l in f:
                if l.rstrip() == '':
                    continue
                self.label_str.append(l.rstrip().split()[0])


    def count_correct_labels(self, log_p, y_gold):
        assert(len(log_p.shape) == 3)
        batch_l, source_l, num_label = log_p.shape
        y_gold = y_gold.contiguous()
        log_p = log_p.contiguous()

        log_p = log_p.view(-1, num_label)  # (batch_l * source_l, num_label)
        y_gold = y_gold.view(-1)            # (batch_l * source_l)

        y_pred = np.argmax(log_p.data, axis=1)  # (batch_l * source_l)
        return np.equal(y_pred, y_gold).sum()


    def get_label(self, log_p, y_gold):
        assert(len(log_p.shape) == 3)
        batch_l, source_l, num_label = log_p.shape
        y_gold = y_gold.contiguous()
        log_p = log_p.contiguous()

        log_p = log_p.view(batch_l, source_l, num_label)  # (batch_l, source_l, num_label)
        y_gold = y_gold.view(batch_l, source_l)         # (batch_l, source_l)

        y_pred = np.argmax(log_p.data, axis=2)  # (batch_l, source_l)

        pred_idx = []
        gold_idx = []
        for ex in y_pred:
            pred_idx.append([self.idx_to_label[int(l)] for l in ex])
        for ex in y_gold:
            gold_idx.append([self.idx_to_label[int(l)] for l in ex])
        return pred_idx, gold_idx

    def get_luka_loss(self, pred, gold):
        # mask = torch.nn.zeros_like(gold)
        correct_class_probs = torch.gather(pred, 2, gold.unsqueeze(-1))
        #correct_class_probs = torch.gather(pred, 2, gold.unsqueeze(-1)).squeeze(-1)
        #seq_probs = torch.prod(correct_class_probs, 1)
        #return -1000000 * torch.sum(seq_probs)

        const = pred.shape[0] * pred.shape[1] - 1

        return (-1 * torch.sum(correct_class_probs)  + const)


    def forward(self, pred, gold):
        batch_l, padded_seq_l, num_label = pred.shape
        log_p = pred.contiguous()
        gold = gold.contiguous()

        assert(num_label == self.opt.num_label + 2)

        # loss
        # crit = torch.nn.NLLLoss(reduction='sum')  # for pytorch < 0.4.1, use size_average=False
        # crit =
        # if self.opt.gpuid != -1:
        #   crit = crit.cuda()

        flat_log_p = log_p.view(-1, num_label)
        flat_gold = gold.view(-1)
        # loss = crit(flat_log_p, flat_gold)
        loss = self.get_luka_loss(pred, gold)
        #print('Luka Loss is ' + str(loss))
        # stats
        batch_l = pred.shape[0]
        padded_seq_l = pred.shape[1]
        # when counting the labels, ignore the <bos> and <eos>
        self.num_correct += self.count_correct_labels(log_p[:, 1:padded_seq_l-1, :], gold[:, 1:padded_seq_l-1])
        self.num_all += batch_l * (padded_seq_l-2)
        self.num_ex += self.shared.batch_l

        # official eval of F1 in NER
        if self.opt.use_f1 == 1:
            pred_label, gold_label = self.get_label(log_p[:, 1:padded_seq_l-1, :], gold[:, 1:padded_seq_l-1])
            self.all_pred_label.extend(pred_label)
            self.all_gold_label.extend(gold_label)


        return loss


    def load_label_map(self):
        idx_to_label = {}
        with open(self.opt.label_dict, 'r') as f:
            for l in f:
                if l.rstrip() == '':
                    continue
                toks = l.rstrip().split()
                idx_to_label[int(toks[1])] = toks[0]
        return idx_to_label


    # return a string of stats
    def print_cur_stats(self):
        stats = 'Acc {0:.3f} '.format(float(self.num_correct) / self.num_all)
        return stats


    # get training metric (scalar metric, extra metric)
    def get_epoch_metric(self):
        acc = float(self.num_correct) / self.num_all

        if self.opt.use_f1 == 1:
            pre, rec, f1 = compute_f1(self.all_pred_label, self.all_gold_label)
            return f1, [pre, rec, f1, acc]
        return acc, [acc]   # and any other scalar metrics


    def begin_pass(self):
        # clear stats
        self.num_correct = 0
        self.num_all = 0
        self.num_ex = 0
        self.all_pred = []

    def end_pass(self):
        pass

