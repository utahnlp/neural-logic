import sys

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
import numpy as np
from official_eval import *


from constraint_utils import get_label_idx, parse_constraint_str
# Godel Loss
class GodelLoss(torch.nn.Module):
    def __init__(self, opt, shared):
        super(GodelLoss, self).__init__()
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

        self.constraints = ['B-VP implies -(I-NP)', 'B-NP implies -(I-VP)', 'O implies -(I-NP)', 'O implies -(I-VP)']
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

    def add_constrained_loss(self, pred, gold, constraint, prod_tnorm='r_prod'):

        label_dict = self.shared.res_map['label_dict']
        # convert pred scorer to probsA

        is_cuda=self.opt.gpuid != -1

        parsed_constraint = parse_constraint_str(constraint)

        left_str, left_multiplier = parsed_constraint[0]
        right_str, right_multiplier = parsed_constraint[2]
        operator = parsed_constraint[1]

        left_index = get_label_idx(label_dict, left_str)
        right_index = get_label_idx(label_dict, right_str)

        left_score = pred[:,:,left_index]
        if left_multiplier == -1:
            left_score = 1 - left_score
        right_score = pred[:,:,right_index]
        if right_multiplier == -1:
            right_score = 1 - right_score

        #left_score_sum = torch.sum(left_score)
        #right_score_sum = torch.sum(right_score)
        #print(constraint)
        #print(left_score)
        #print(right_score)
        #print(pred_prob)
        #print(pred.shape)
        if operator == 'subtract':
            loss_value = -1 * torch.sum(torch.max(1-left_score, right_score))
            return loss_value
        elif operator == 'add':
            return (left_score_sum * left_multiplier) + (right_multiplier * right_score_sum)
        else:
            raise NotImplementedError('Operator not implemented')

    def get_godel_loss(self, pred, gold):
        correct_class_probs = torch.gather(pred, 2, gold.unsqueeze(-1))

        return (-1 * torch.min(correct_class_probs))


    def forward(self, pred, gold, constraints_lambda=0.001):
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
        loss = self.get_godel_loss(pred, gold)
        constraint_loss = 0.0

        for constraint in self.constraints:
            constraint_loss += self.add_constrained_loss(pred, gold, constraint)

        #print(loss, constraints_lambda * constraint_loss)
        loss = loss + constraints_lambda * constraint_loss

        #print('Godel Loss is ' + str(loss))
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

