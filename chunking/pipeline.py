import sys
#sys.path.insert(0, './constraint')

import torch
import torch.nn as nn
from torch.autograd import Variable as Var
from char_embeddings import *
from embeddings import *
from encoder import *
from classifier import *
from crf import *
#from s1 import *
#from s2 import *
#from s3 import *
#from s4 import *
#from s5 import *
#from s6 import *
#from s7 import *
#from n1 import *
#from n2 import *
#from n3 import *
#from n4 import *
#from n7 import *
#from s1to4 import *

class Pipeline(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Pipeline, self).__init__()
		self.opt = opt
		self.shared = shared

		rnn_hidden_size = opt.hidden_size if opt.bidir == 0 else opt.hidden_size/2

		self.logic_layers = self.get_logic_layers(opt.constr)
		self.rhos = []
		if opt.rhos.strip() != '':
			for tok in opt.rhos.split(','):
				rho = Variable(torch.ones(1) * float(tok), requires_grad=False)
				if opt.gpuid != -1:
					rho = rho.cuda()
				self.rhos.append(rho)

		# architecture
		if opt.use_char_enc == 1:
			self.char_embeddings = CharEmbeddings(opt, shared)

		self.embeddings = WordVecLookup(opt, shared)
		self.encoder = Encoder(opt, shared)
		self.classifier = Classifier(opt, shared)

		if opt.use_crf == 1:
			self.crf = CRF(opt, shared)

		self.log_softmax = nn.LogSoftmax(2)

		# Ashim : softmax declaration
		self.softmax = nn.Softmax(2)

	def forward(self, sent, char_sent):

		if self.opt.use_char_enc == 1:
			char_sent = self.char_embeddings(char_sent)   # (batch_l, context_l, token_l, char_emb_size)
		else:
			char_sent = None

		sent = self.embeddings(sent)


		sent = self.encoder(sent, char_sent)
		assert(sent.shape == (self.shared.batch_l, self.shared.source_l, self.opt.hidden_size))

		y_score = self.classifier(sent)

		y_score = self.logic_layer(y_score)

		out = None
		if self.opt.use_crf == 1:
			self.shared.crf = self.crf
			self.shared.crf_partition = self.crf(y_score)
			out = y_score
		else:
			if self.opt.use_luka or self.opt.use_godel:
				# Ashim : For Lukasevic Loss (Use Probabilities, take softmax)
				out = self.softmax(y_score)
			else:
				# prepare for nll loss # out : log probabilities
				out = self.log_softmax(y_score)
		return out


	def logic_layer(self, scores):
		assert(len(self.rhos) == len(self.logic_layers))
		if len(self.logic_layers) == 0:
			return scores

		score_prime_ls = []
		for l, rho in zip(self.logic_layers, self.rhos):
			score_prime_ls.append(l(scores, rho).unsqueeze(0))

		return torch.cat(score_prime_ls, 0).sum(0)


	def update_context(self, batch_idx, batch_l, source_l, res_map):
		self.shared.batch_idx = batch_idx
		self.shared.batch_l = batch_l
		self.shared.source_l = source_l
		self.shared.res_map = res_map


	def get_logic_layers(self, names):
		layers = []

		if names == '':
			return layers

		for n in names.split(','):
			if n == 's1':
				layers.append(S1(self.opt, self.shared))
			elif n == 's2':
				layers.append(S2(self.opt, self.shared))
			elif n == 's3':
				layers.append(S3(self.opt, self.shared))
			elif n == 's4':
				layers.append(S4(self.opt, self.shared))
			elif n == 's1to4':
				layers.append(S1to4(self.opt, self.shared))
			elif n == 's5':
				layers.append(S5(self.opt, self.shared))
			elif n == 's6':
				layers.append(S6(self.opt, self.shared))
			elif n == 's7':
				layers.append(S7(self.opt, self.shared))
			elif n == 'n1':
				layers.append(N1(self.opt, self.shared))
			elif n == 'n2':
				layers.append(N2(self.opt, self.shared))
			elif n == 'n3':
				layers.append(N3(self.opt, self.shared))
			elif n == 'n4':
				layers.append(N4(self.opt, self.shared))
			elif n == 'n7':
				layers.append(N7(self.opt, self.shared))
			else:
				print('unrecognized constraint layer name: {0}'.format(n))
				assert(False)
		return layers


	def init_weight(self):
		missed_names = []
		if self.opt.param_init_type == 'xavier_uniform':
			for n, p in self.named_parameters():
				if p.requires_grad and not hasattr(p, 'skip_init'):
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_uniform_(p)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant_(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.opt.param_init_type == 'xavier_normal':
			for n, p in self.named_parameters():
				if p.requires_grad and not hasattr(p, 'skip_init'):
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_normal_(p)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant_(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.opt.param_init_type == 'no':
			for n, p in self.named_parameters():
				missed_names.append(n)
		else:
			assert(False)

		if len(missed_names) != 0:
			print('uninitialized fields: {0}'.format(missed_names))

	def begin_pass(self):
		if self.opt.use_char_enc == 1:
			self.char_embeddings.begin_pass()
		self.embeddings.begin_pass()
		self.encoder.begin_pass()
		self.classifier.begin_pass()
		if self.opt.use_crf == 1:
			self.crf.begin_pass()


	def end_pass(self):
		if self.opt.use_char_enc == 1:
			self.char_embeddings.end_pass()
		self.embeddings.end_pass()
		self.encoder.end_pass()
		self.classifier.end_pass()
		if self.opt.use_crf == 1:
			self.crf.end_pass()


	def get_param_dict(self):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		skipped_fields = []
		for n, p in self.named_parameters():
			# save all parameters that do not have skip_save flag
			#   unlearnable parameters will also be saved
			if not hasattr(p, 'skip_save') or p.skip_save == 0:
				param_dict[n] =  torch2np(p.data, is_cuda)
			else:
				skipped_fields.append(n)
		#print('skipped fields:', skipped_fields)
		return param_dict


	def set_param_dict(self, param_dict):
		skipped_fields = []
		rec_fields = []
		for n, p in self.named_parameters():
			if n in param_dict:
				rec_fields.append(n)
				# load everything we have
				print('setting {0}'.format(n))
				p.data.copy_(torch.from_numpy(param_dict[n][:]))
			else:
				skipped_fields.append(n)
		print('skipped fileds: {0}'.format(skipped_fields))

