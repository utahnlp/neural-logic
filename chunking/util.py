import sys
import h5py
import torch
from torch import cuda
from torch import nn

def torch2np(t, is_cuda):
	return t.numpy() if not is_cuda else t.cpu().numpy()


def rand_tensor(shape, r1, r2):
	return (r1 - r2) * torch.rand(shape) + r2


def save_opt(opt, path):
	with open(path, 'w') as f:
		f.write('{0}'.format(opt))


def load_param_dict(path):
	# TODO, this is ugly
	f = h5py.File(path, 'r')
	return f


def save_param_dict(param_dict, path):
	file = h5py.File(path, 'w')
	for name, p in param_dict.items():
		file.create_dataset(name, data=p)

	file.close()


def load_dict(path):
	rs = {}
	with open(path, 'r+') as f:
		for l in f:
			if l.strip() == '':
				continue
			w, idx, cnt = l.strip().split()
			rs[int(idx)] = w
	return rs



def build_rnn(type, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional):
	if type == 'lstm':
		return nn.LSTM(input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			bias=bias,
			batch_first=batch_first,
			dropout=dropout,
			bidirectional=bidirectional)
	elif type == 'gru':
		return nn.GRU(input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			bias=bias,
			batch_first=batch_first,
			dropout=dropout,
			bidirectional=bidirectional)
	else:
		assert(False)