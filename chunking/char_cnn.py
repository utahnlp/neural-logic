import torch
from torch import nn
from torch.autograd import Variable
from holder import *


class CharCNN(torch.nn.Module):
	def __init__(self, opt, shared):
		super(CharCNN, self).__init__()
		self.opt = opt
		self.shared = shared

		filter_sizes = [int(i) for i in opt.char_filters.strip().split(',')]

		self.conv_layers = []
		for k_size in filter_sizes:
			# C_i is char_emb_size
			# C_o is char_enc_size
			# kernel is (1, k_size)
			conv = nn.Conv2d(opt.char_emb_size, opt.char_enc_size, (1, k_size))
			self.conv_layers.append(conv)
		self.conv_layers = nn.ModuleList(self.conv_layers)

		post_conv_size = opt.char_enc_size * len(filter_sizes)
		if post_conv_size != opt.char_enc_size:
			self.proj = nn.Linear(post_conv_size, opt.char_enc_size)

		self.bias = nn.Parameter(torch.zeros(len(filter_sizes)), requires_grad=True)
		self.bias.skip_init = 1

		self.activation = nn.ReLU()
		self.drop = nn.Dropout(0.2)
		

	# input size (batch_l * seq_l, token_l, char_emb_size)
	# output size (batch_l * seq_l, char_enc_size)
	def forward(self, x):
		# transform input to shape (batch_l * seq_l, char_emb_size, 1, token_l)
		x = self.drop(x)
		x = x.transpose(1,2).unsqueeze(2)

		# output of conv is (batch_l * seq_l, char_enc_size, 1, *)
		#	squeeze the 3rd dim and max over the last
		cnn_encs = [self.activation(conv(x)).squeeze(2) + self.bias[i] for i, conv in enumerate(self.conv_layers)]
		max_out = [enc.max(-1)[0] for enc in cnn_encs]	# (batch_l * seq_l, char_enc_size)

		# concat into one vector for each example
		phi = torch.cat(max_out, 1)	# (batch_l * seq_l, char_enc_size * |char_filters|)

		# if need to downsample to certain dim
		if hasattr(self, 'proj'):
			phi = self.proj(phi)

		return phi


