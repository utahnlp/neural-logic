import sys

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
import numpy as np
from official_eval import *

from constraint_utils import get_label_idx, parse_constraint_str


# NLL Loss
class Loss(torch.nn.Module):
    def __init__(self, opt, shared):
        super(Loss, self).__init__()
        self.opt = opt
        self.shared = shared
        # do not creat loss node globally
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
        self.softmax = nn.Softmax(2)

        self.constraints = ['B-NP implies -(I-VP)', 'B-NP implies -(I-ADVP)', 'B-NP implies -(I-ADJP)', 'B-NP implies -(I-SBAR)', 'B-NP implies -(I-PP)', 'B-NP implies -(I-PRT)', 'B-NP implies -(I-LST)', 'B-NP implies -(I-CONJP)', 'B-NP implies -(I-UCP)', 'B-NP implies -(I-INTJ)', 'B-VP implies -(I-NP)', 'B-VP implies -(I-ADVP)', 'B-VP implies -(I-ADJP)', 'B-VP implies -(I-SBAR)', 'B-VP implies -(I-PP)', 'B-VP implies -(I-PRT)', 'B-VP implies -(I-LST)', 'B-VP implies -(I-CONJP)', 'B-VP implies -(I-UCP)', 'B-VP implies -(I-INTJ)', 'B-ADVP implies -(I-NP)', 'B-ADVP implies -(I-VP)', 'B-ADVP implies -(I-ADJP)', 'B-ADVP implies -(I-SBAR)', 'B-ADVP implies -(I-PP)', 'B-ADVP implies -(I-PRT)', 'B-ADVP implies -(I-LST)', 'B-ADVP implies -(I-CONJP)', 'B-ADVP implies -(I-UCP)', 'B-ADVP implies -(I-INTJ)', 'B-ADJP implies -(I-NP)', 'B-ADJP implies -(I-VP)', 'B-ADJP implies -(I-ADVP)', 'B-ADJP implies -(I-SBAR)', 'B-ADJP implies -(I-PP)', 'B-ADJP implies -(I-PRT)', 'B-ADJP implies -(I-LST)', 'B-ADJP implies -(I-CONJP)', 'B-ADJP implies -(I-UCP)', 'B-ADJP implies -(I-INTJ)', 'B-SBAR implies -(I-NP)', 'B-SBAR implies -(I-VP)', 'B-SBAR implies -(I-ADVP)', 'B-SBAR implies -(I-ADJP)', 'B-SBAR implies -(I-PP)', 'B-SBAR implies -(I-PRT)', 'B-SBAR implies -(I-LST)', 'B-SBAR implies -(I-CONJP)', 'B-SBAR implies -(I-UCP)', 'B-SBAR implies -(I-INTJ)', 'B-PP implies -(I-NP)', 'B-PP implies -(I-VP)', 'B-PP implies -(I-ADVP)', 'B-PP implies -(I-ADJP)', 'B-PP implies -(I-SBAR)', 'B-PP implies -(I-PRT)', 'B-PP implies -(I-LST)', 'B-PP implies -(I-CONJP)', 'B-PP implies -(I-UCP)', 'B-PP implies -(I-INTJ)', 'B-PRT implies -(I-NP)', 'B-PRT implies -(I-VP)', 'B-PRT implies -(I-ADVP)', 'B-PRT implies -(I-ADJP)', 'B-PRT implies -(I-SBAR)', 'B-PRT implies -(I-PP)', 'B-PRT implies -(I-LST)', 'B-PRT implies -(I-CONJP)', 'B-PRT implies -(I-UCP)', 'B-PRT implies -(I-INTJ)', 'B-LST implies -(I-NP)', 'B-LST implies -(I-VP)', 'B-LST implies -(I-ADVP)', 'B-LST implies -(I-ADJP)', 'B-LST implies -(I-SBAR)', 'B-LST implies -(I-PP)', 'B-LST implies -(I-PRT)', 'B-LST implies -(I-CONJP)', 'B-LST implies -(I-UCP)', 'B-LST implies -(I-INTJ)', 'B-CONJP implies -(I-NP)', 'B-CONJP implies -(I-VP)', 'B-CONJP implies -(I-ADVP)', 'B-CONJP implies -(I-ADJP)', 'B-CONJP implies -(I-SBAR)', 'B-CONJP implies -(I-PP)', 'B-CONJP implies -(I-PRT)', 'B-CONJP implies -(I-LST)', 'B-CONJP implies -(I-UCP)', 'B-CONJP implies -(I-INTJ)', 'B-UCP implies -(I-NP)', 'B-UCP implies -(I-VP)', 'B-UCP implies -(I-ADVP)', 'B-UCP implies -(I-ADJP)', 'B-UCP implies -(I-SBAR)', 'B-UCP implies -(I-PP)', 'B-UCP implies -(I-PRT)', 'B-UCP implies -(I-LST)', 'B-UCP implies -(I-CONJP)', 'B-UCP implies -(I-INTJ)', 'B-INTJ implies -(I-NP)', 'B-INTJ implies -(I-VP)', 'B-INTJ implies -(I-ADVP)', 'B-INTJ implies -(I-ADJP)', 'B-INTJ implies -(I-SBAR)', 'B-INTJ implies -(I-PP)', 'B-INTJ implies -(I-PRT)', 'B-INTJ implies -(I-LST)', 'B-INTJ implies -(I-CONJP)', 'B-INTJ implies -(I-UCP)']
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
        pred_prob = self.softmax(pred)
        #Not actually a dict

        is_cuda=self.opt.gpuid != -1

        parsed_constraint = parse_constraint_str(constraint)

        left_str, left_multiplier = parsed_constraint[0]
        right_str, right_multiplier = parsed_constraint[2]
        operator = parsed_constraint[1]

        left_index = get_label_idx(label_dict, left_str)
        right_index = get_label_idx(label_dict, right_str)

        left_score = pred_prob[:,:,left_index]
        if left_multiplier == -1:
            left_score = 1 - left_score
        right_score = pred_prob[:,:,right_index]
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
            if prod_tnorm == 's_prod':
                elements = (1 - left_score + left_score*right_score) + 0.00001
            else:
                division = right_score/(left_score+0.001)
                ones_tensor = torch.ones(division.shape).cuda()
                #print(left_score, right_score)
                #print(right_score/left_score)
                #print(left_score/right_score)
                elements = torch.min(ones_tensor, division)
                #elements = torch.clamp(right_score/left_score, min=0.00001, max=1.0)
                #print(elements)
                #elements = (1 - left_score + left_score*right_score) + 0.00001
                #print(elements)

            loss_value = -1 * torch.sum(torch.log(elements))
            return loss_value
        elif operator == 'add':
            return (left_score_sum * left_multiplier) + (right_multiplier * right_score_sum)
        else:
            raise NotImplementedError('Operator not implemented')

    def forward(self, pred, gold, constraints_lambda=0.001):
        batch_l, padded_seq_l, num_label = pred.shape
        log_p = pred.contiguous()
        gold = gold.contiguous()

        assert(num_label == self.opt.num_label + 2)

        # loss
        crit = torch.nn.NLLLoss(reduction='sum')    # for pytorch < 0.4.1, use size_average=False
        if self.opt.gpuid != -1:
            crit = crit.cuda()

        flat_log_p = log_p.view(-1, num_label)
        flat_gold = gold.view(-1)
        loss = crit(flat_log_p, flat_gold)

        constraint_loss = 0.0

        for constraint in self.constraints:
            constraint_loss += self.add_constrained_loss(pred, gold, constraint)
        cr_loss = loss
        loss = loss + constraints_lambda * constraint_loss
        #if constraint_loss.detach().cpu().item() != 0.0:
        #    print('Not zero')
        #    print(cr_loss, constraint_loss)
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

