import sys

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *


class CRF(torch.nn.Module):
	def __init__(self, opt, shared):
		super(CRF, self).__init__()
		self.opt = opt
		self.shared = shared

		self.trans_weight = nn.Parameter(
			torch.ones(opt.num_label+2, opt.num_label+2), requires_grad=True)
		# hacky, to postpone initialization to make sure data split are the same as basline
		self.trans_weight.skip_init = 1
		self.trans_weight.initialized = 0

		self.bos_idx = 0
		self.eos_idx = 1
		self.trans_weight.data[self.bos_idx, :] = -10000.0
		self.trans_weight.data[:, self.eos_idx] = -10000.0

	def __init_trans(self):
		if self.trans_weight.requires_grad and self.training:
			print('lazy initializing transition weight')
			#nn.init.xavier_uniform_(self.trans_weight)
			nn.init.normal(self.trans_weight, 0, 1)
			#self.trans_weight.data[self.bos_idx, :] = -10000.0
			#self.trans_weight.data[:, self.eos_idx] = -10000.0

	def log_sum_exp(self, vec, dim=0):
		max_v, idx = torch.max(vec, dim)
		max_exp = max_v.unsqueeze(-1).expand_as(vec)
		return max_v + torch.log(torch.sum(torch.exp(vec - max_exp), dim))

	def argmax(self, x): # for 1D tensor
		return torch.max(x, 0)[1].data[0]

	# get the partition Z
	# score of shape (batch_l, source_l, num_label+2)
	def forward(self, score):
		# trim off the <box> and <eos>
		score = score[:, 1:-1, :]
		batch_size, seq_len, n_labels = score.size()
		alpha = score.data.new(batch_size, n_labels).fill_(-10000)
		alpha[:, self.bos_idx] = 0
		alpha = Variable(alpha)
		lens = Variable(torch.LongTensor([seq_len]*batch_size))
		if self.opt.gpuid != -1:
			alpha = alpha.cuda()
			lens = lens.cuda()

		c_lens = lens.clone()

		logits_t = score.transpose(1, 0)
		for logit in logits_t:
			logit_exp = logit.unsqueeze(-1).expand(batch_size,
												   *self.trans_weight.size())
			alpha_exp = alpha.unsqueeze(1).expand(batch_size,
												  *self.trans_weight.size())
			trans_exp = self.trans_weight.unsqueeze(0).expand_as(alpha_exp)
			mat = trans_exp + alpha_exp + logit_exp
			alpha_nxt = self.log_sum_exp(mat, 2).squeeze(-1)

			mask = (c_lens > 0).float().unsqueeze(-1).expand_as(alpha)
			alpha = mask * alpha_nxt + (1 - mask) * alpha
			c_lens = c_lens - 1

		alpha = alpha + self.trans_weight[self.eos_idx].unsqueeze(0).expand_as(alpha)
		norm = self.log_sum_exp(alpha, 1).squeeze(-1)

		return norm

	# viterbi decoding
	# 	input y_score of shape (batch_l, source_l, num_label+2)
	def viterbi_decode(self, y_score):
		# trim off the <box> and <eos>
		y_score = y_score[:, 1:-1, :]
		batch_size, seq_len, n_labels = y_score.size()
		vit = y_score.data.new(batch_size, n_labels).fill_(-10000)
		vit[:, self.bos_idx] = 0
		vit = Variable(vit)
		lens = Variable(torch.LongTensor([seq_len]*batch_size))
		if self.opt.gpuid != -1:
			vit = vit.cuda()
			lens = lens.cuda()

		c_lens = lens.clone()

		logits_t = y_score.transpose(1, 0)
		pointers = []
		for logit in logits_t:
			vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
			trn_exp = self.trans_weight.unsqueeze(0).expand_as(vit_exp)
			vit_trn_sum = vit_exp + trn_exp
			vt_max, vt_argmax = vit_trn_sum.max(2)

			vt_max = vt_max.squeeze(-1)
			vit_nxt = vt_max + logit
			pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

			mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
			vit = mask * vit_nxt + (1 - mask) * vit

			mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
			vit += mask * self.trans_weight[ self.eos_idx ].unsqueeze(0).expand_as(vit_nxt)

			c_lens = c_lens - 1

		pointers = torch.cat(pointers)
		scores, idx = vit.max(1)
		idx = idx.squeeze(-1)
		if len(idx.shape) == 0:
			idx = idx.view(1)
		paths = [idx.unsqueeze(1)]

		for argmax in reversed(pointers):
			idx_exp = idx.unsqueeze(-1)
			idx = torch.gather(argmax, 1, idx_exp)
			idx = idx.squeeze(-1)

			paths.insert(0, idx.unsqueeze(1))

		paths = torch.cat(paths[1:], 1)
		scores = scores.squeeze(-1)

		# reconcat
		bos = Variable(torch.LongTensor([self.bos_idx]*batch_size)).view(batch_size, 1)
		eos = Variable(torch.LongTensor([self.eos_idx]*batch_size)).view(batch_size, 1)
		if self.opt.gpuid != -1:
			bos = bos.cuda()
			eos = eos.cuda()
		paths = torch.cat([bos, paths, eos], -1)

		return scores, paths


	def begin_pass(self):
		if self.trans_weight.initialized == 0:
			self.__init_trans()
			self.trans_weight.initialized = 1
		

	def end_pass(self):
		pass



if __name__ == '__main__':
	opt = Holder()
	shared = Holder()
	opt.gpuid = -1
	opt.num_label = 5
	shared.batch_l = 1
	shared.source_l = 10

	y_score = Variable(torch.randn(shared.batch_l, shared.source_l, opt.num_label+2))
	crf = CRF(opt, shared)

	z = crf(y_score)
	print(z)
	y = crf.viterbi_decode(y_score)
	print(y)