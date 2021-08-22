import h5py
import torch
from torch import nn
from torch import cuda
import numpy as np
#import ujson
from util import *

class Data():
	def __init__(self, opt, data_file, res_files=None):
		self.opt = opt
		self.data_name = data_file

		print('loading data from {0}'.format(data_file))
		f = h5py.File(data_file, 'r')
		self.source = f['source'][:]
		self.all_source = f['all_source'][:]
		self.label = f['label'][:]
		self.source_l = f['source_l'][:]	# (num_batch,)
		self.batch_l = f['batch_l'][:]
		self.batch_idx = f['batch_idx'][:]
		self.ex_idx = f['ex_idx'][:]
		self.length = self.batch_l.shape[0]

		print('loading char idx from {0}'.format(opt.char_idx))
		f = h5py.File(opt.char_idx, 'r')
		self.char_idx = f['char_idx'][:]
		self.char_idx = torch.from_numpy(self.char_idx)
		assert(self.char_idx.shape[1] == opt.token_l)
		assert(self.char_idx.max()+1 == opt.num_char)
		print('{0} chars found'.format(self.char_idx.max()+1))

		self.all_source = torch.from_numpy(self.all_source)
		self.source = torch.from_numpy(self.source)
		self.label = torch.from_numpy(self.label)

		if self.opt.gpuid != -1:
			self.source = self.source.cuda()
			self.label = self.label.cuda()

		self.batches = []
		for i in range(self.length):
			start = self.batch_idx[i]
			end = start + self.batch_l[i]

			# get example token indices
			all_source_i = self.all_source[start:end, 0:self.source_l[i]]
			source_i = self.source[start:end, 0:self.source_l[i]]
			label_i = self.label[start:end, 0:self.source_l[i]]

			# sanity check
			assert(self.source[start:end, self.source_l[i]:].sum() == 0)
			assert(self.label[start:end, self.source_l[i]:].sum() == 0)

			# src, tgt, batch_l, src_l, tgt_l, span, raw info
			self.batches.append((source_i, all_source_i, label_i, int(self.batch_l[i]), int(self.source_l[i])))

		# count examples
		self.num_ex = 0
		for i in range(self.length):
			self.num_ex += self.batch_l[i]

		# load resource files
		self.res_names = []
		if res_files is not None:
			for f in res_files:
				print('loading res file from {0}...'.format(f))
				res_name = self.__load_res(f)
				# record the name
				self.res_names.append(res_name)

		print('loading label dict from : {0}'.format(opt.label_dict))
		self.__load_label_dict(opt.label_dict)


	def subsample(self, ratio, random):
		target_num_ex = int(float(self.num_ex) * ratio)
		sub_idx = [int(idx) for idx in torch.randperm(self.size())] if random else [i for i in range(self.size())]
		cur_num_ex = 0
		i = 0
		while cur_num_ex < target_num_ex:
			cur_num_ex += self.batch_l[sub_idx[i]]
			i += 1
		return sub_idx[:i], cur_num_ex


	def split(self, sub_idx, ratio):
		num_ex = sum([self.batch_l[i] for i in sub_idx])
		target_num_ex = int(float(num_ex) * ratio)

		cur_num_ex = 0
		cur_pos = 0
		for i in range(len(sub_idx)):
			cur_pos = i
			cur_num_ex += self.batch_l[sub_idx[i]]
			if cur_num_ex >= target_num_ex:
				break

		return sub_idx[:cur_pos+1], sub_idx[cur_pos+1:], cur_num_ex, num_ex - cur_num_ex


	def __load_label_dict(self, path):
		d = []
		with open(path, 'r+') as f:
			for l in f:
				if l.strip() == '':
					continue
				d.append(l.strip().split()[0])
		assert(len(d) == self.opt.num_label + 2)
		self.label_dict = d


	def __load_res(self, f):
		if '.pos.txt' in f:
			return self.__load_pos(f)
		else:
			print('unrecognized resource file: {0}'.format(f))
			assert(False)


	def __load_pos(self, pos_path):
		rs = []
		with open(pos_path, 'r+') as f:
			for l in f:
				if l.strip() == '':
					continue
				pos_tags = l.strip().split(' ')
				rs.append(['<bos>'] + pos_tags + ['<eos>'])
		self.pos = rs
		return 'pos'


	def size(self):
		return self.length


	def __getitem__(self, idx):
		source, all_source, label, batch_l, source_l = self.batches[idx]
		token_l = self.opt.token_l

		# get batch ex indices
		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]
		char_source = self.char_idx[all_source.contiguous().view(-1)].view(batch_l, source_l, token_l)

		if self.opt.gpuid != -1:
			char_source = char_source.cuda()

		res_map = self.__get_res(idx)

		# append label dict as well
		if res_map is None:
			res_map = {}
		res_map['label_dict'] = self.label_dict

		return (self.data_name, source, char_source, batch_ex_idx, batch_l, source_l, label, res_map)


	def __get_res(self, idx):
		# if there is no resource presents, return None
		if len(self.res_names) == 0:
			return None


		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		all_res = {}
		for res_n in self.res_names:
			res = getattr(self, res_n)

			batch_res = [res[ex_id] for ex_id in batch_ex_idx]
			all_res[res_n] = batch_res

		return all_res

