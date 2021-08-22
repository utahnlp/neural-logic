import sys
#sys.path.insert(0, '../')

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *


def get_label_idx(labels, key):
    for i, l in enumerate(labels):
        if l.startswith(key):
            return i

    raise ValueError("Label key {0} not present".format(str(key)))


def parse_constraint_str(s):
    # Parse the constraint string to get left atom, right atom,
    # and operands
    # B-NP implies -(I-VP) : (p1 + p2)
    parts = s.strip().split()
    left_multiplier = 1
    if '(' in parts[0]:
        left_multiplier = -1
        left_str = parts[0][2:-1]
    else:
        left_multiplier = 1
        left_str = parts[0]

    if '(' in parts[2]:
        right_multiplier = -1
        right_str = parts[2][2:-1]
    else:
        right_multiplier = 1
        right_str = parts[2]


    if 'implies' in parts[1]:
        operator = 'subtract'
    else:
        raise NotImplementedError('Logical operator not implemented')
        #opertor = 'add'

    return (left_str, left_multiplier), operator, (right_str, right_multiplier)



def build_mask(labels, key, is_cuda):
    mask = Variable(
        torch.Tensor([float(l.startswith(key)) for l in labels]).view(1, len(labels)),
        requires_grad=False)

    if is_cuda:
        mask = mask.cuda()
    return mask
