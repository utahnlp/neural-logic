import sys

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
import numpy as np
from crf import *
from official_eval import *

# NLL of CRF loss
class CRFLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(CRFLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		# do not creat loss node globally
		self.num_correct = 0
		self.num_all = 0
		self.num_ex = 0
		self.verbose = False

		self.bos_idx = 0
		self.eos_idx = 1

		# for official eval of f1 in NER
		self.all_pred_label = []
		self.all_gold_label = []
		self.idx_to_label = self.load_label_map()


	def count_correct_labels_from_score(self, log_p, y_gold):
		assert(len(log_p.shape) == 3)
		batch_l, source_l, num_label = log_p.shape
		y_gold = y_gold.contiguous()
		log_p = log_p.contiguous()
	
		log_p = log_p.view(-1, num_label)  # (batch_l * source_l, num_label)
		y_gold = y_gold.view(-1)			# (batch_l * source_l)
	
		y_pred = np.argmax(log_p.data, axis=1)	# (batch_l * source_l)
		return np.equal(y_pred, y_gold).sum()


	def count_correct_labels_from_idx(self, y_pred, y_gold):
		assert(len(y_pred.shape) == 2)
		y_pred = y_pred.contiguous()
		y_gold = y_gold.contiguous()
		return np.equal(y_pred, y_gold).sum()


	def score_to_idx(self, log_p):
		log_p = log_p.contiguous()
		return np.argmax(log_p.data, axis=2)	# (batch_l, source_l)


	# gold of shape (batch_l, source_l)
	def score_gold_transition(self, gold):
		gold = gold[:, 1:-1]

		batch_size, seq_len = gold.size()
		lens = Variable(torch.LongTensor([seq_len]*batch_size))
		if self.opt.gpuid != -1:
			lens = lens.cuda()

		# pad labels with <start> and <stop> indices
		labels_ext = Variable(gold.data.new(batch_size, seq_len + 2))
		labels_ext[:, 0] = self.bos_idx
		labels_ext[:, 1:-1] = gold
		mask = self.sequence_mask(lens + 1, max_len=seq_len + 2).long()
		pad_stop = Variable(gold.data.new(1).fill_(self.eos_idx))
		pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
		labels_ext = (1 - mask) * pad_stop + mask * labels_ext
		labels = labels_ext

		trn = self.shared.crf.trans_weight

		# obtain transition vector for each label in batch and timestep
		# (except the last ones)
		trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
		lbl_r = labels[:, 1:]
		lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
		trn_row = torch.gather(trn_exp, 1, lbl_rexp)

		# obtain transition score from the transition vector for each label
		# in batch and timestep (except the first ones)
		lbl_lexp = labels[:, :-1].unsqueeze(-1)
		trn_scr = torch.gather(trn_row, 2, lbl_lexp)
		trn_scr = trn_scr.squeeze(-1)

		mask = self.sequence_mask(lens + 1).float()
		trn_scr = trn_scr * mask
		score = trn_scr.sum(1).squeeze(-1)

		return score

	def score_gold_emission(self, y_score, gold):
		y_score = y_score[:, 1:-1, :]
		gold = gold[:, 1:-1]

		batch_size, seq_len = gold.size()
		lens = Variable(torch.LongTensor([seq_len]*batch_size))
		if self.opt.gpuid != -1:
			lens = lens.cuda()

		y_exp = gold.unsqueeze(-1)
		scores = torch.gather(y_score, 2, y_exp).squeeze(-1)
		mask = self.sequence_mask(lens).float()
		scores = scores * mask
		score = scores.sum(1).squeeze(-1)

		return score


	def score_gold_crf(self, y_score, gold):
		trans = self.score_gold_transition(gold)
		emit = self.score_gold_emission(y_score, gold)
		return trans + emit


	def sequence_mask(self, lens, max_len=None):
		batch_size = lens.size(0)
	
		if max_len is None:
			max_len = lens.max().data[0]
	
		ranges = torch.arange(0, max_len).long()
		ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
		ranges = Variable(ranges)
	
		if lens.data.is_cuda:
			ranges = ranges.cuda()
	
		lens_exp = lens.unsqueeze(1).expand_as(ranges)
		mask = ranges < lens_exp
	
		return mask


	# the input pred should be the original score (NOT logsoftmax!!!)
	#	pred of shape (batch_l, source_l, num_label+2)
	#	gold of shape (batch_l, source_l)
	def forward(self, pred, gold):
		batch_l, padded_seq_l, num_label = pred.shape
		y_score = pred.contiguous()
		y_gold = gold.contiguous()

		assert(num_label == self.opt.num_label + 2)

		# loss
		gold_score = self.score_gold_crf(y_score, y_gold)
		loss = (self.shared.crf_partition - gold_score).sum()

		# stats
		batch_l = pred.shape[0]
		padded_seq_l = pred.shape[1]
		# when counting the labels, ignore the <bos> and <eos>
		# in crf mode, use the viterbi output as prediction
		if self.opt.use_crf == 1 and not self.training:
			crf_score, crf_pred = self.shared.crf.viterbi_decode(y_score)
			self.num_correct += self.count_correct_labels_from_idx(crf_pred[:, 1:padded_seq_l-1], y_gold[:, 1:padded_seq_l-1])
			# official eval of F1 in NER
			if self.opt.use_f1 == 1:
				pred_label, gold_label = self.get_label(crf_pred[:, 1:padded_seq_l-1], y_gold[:, 1:padded_seq_l-1])
				self.all_pred_label.extend(pred_label)
				self.all_gold_label.extend(gold_label)
		else:
			y_pred = nn.LogSoftmax(2)(y_score)
			self.num_correct += self.count_correct_labels_from_score(y_pred[:, 1:padded_seq_l-1, :], y_gold[:, 1:padded_seq_l-1])
			# official eval of F1 in NER
			if self.opt.use_f1 == 1:
				pred_label, gold_label = self.get_label(self.score_to_idx(y_pred[:, 1:padded_seq_l-1]), y_gold[:, 1:padded_seq_l-1])
				self.all_pred_label.extend(pred_label)
				self.all_gold_label.extend(gold_label)

		self.num_all += batch_l * (padded_seq_l-2)
		self.num_ex += self.shared.batch_l

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


	def get_label(self, y_pred, y_gold):
		pred_idx = []
		gold_idx = []
		for ex in y_pred:
			pred_idx.append([self.idx_to_label[int(l)] for l in ex])
		for ex in y_gold:
			gold_idx.append([self.idx_to_label[int(l)] for l in ex])
		return pred_idx, gold_idx


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
		return acc, [acc] 	# and any other scalar metrics	



	def begin_pass(self):
		# clear stats
		self.num_correct = 0
		self.num_all = 0
		self.num_ex = 0

	def end_pass(self):
		pass

if __name__ == '__main__':
	#torch.manual_seed(1)
	from crf import *
	opt = Holder()
	shared = Holder()
	opt.gpuid = -1
	opt.num_label = 3
	shared.batch_l = 3
	shared.source_l = 10

	y_score = Variable(torch.randn(shared.batch_l, shared.source_l, opt.num_label+2))
	crf = CRF(opt, shared)
	crf_loss = CRFLoss(opt, shared)

	shared.crf = crf

	y_gold = Variable(torch.LongTensor([[0,0,1,2,3,4,1,2,3,1], [0,0,1,2,3,4,1,2,3,1], [0,0,1,2,3,4,1,2,3,1]]))
	y = crf_loss.score_gold_crf(y_score, y_gold)
	print(y)


