import sys

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *


class Classifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Classifier, self).__init__()
		self.opt = opt
		self.shared = shared

		self.linear = nn.Sequential(
			nn.Linear(opt.hidden_size, opt.num_label+2))


	# input x of shape (batch_l, seq_l, hidden_size)
	def forward(self, x):
		x = x.contiguous()
		batch_l, seq_l, hidden_size = x.shape
		return self.linear(x.view(batch_l * seq_l, hidden_size)).view(batch_l, seq_l, self.opt.num_label+2)

	def begin_pass(self):
		pass

	def end_pass(self):
		pass
