import argparse
import sys
import numpy as np
import h5py

class Indexer:
	def __init__(self, symbols = ["<blank>", "<bos>", '<eos>'], num_oov=1):
		self.d = {}
		self.cnt = {}
		for s in symbols:
			self.d[s] = len(self.d)
			self.cnt[s] = 0

		self.num_oov = num_oov
		
		for i in range(self.num_oov): #hash oov words to one of 100 random embeddings
			oov_word = '<oov'+ str(i) + '>'
			self.d[oov_word] = len(self.d)
			self.cnt[oov_word] = 10000000   # have a large number for oov word to avoid being pruned
			
	def convert(self, w):		
		return self.d[w] if w in self.d else self.d['<oov' + str(np.random.randint(self.num_oov)) + '>']

	def convert_sequence(self, ls):
		return [self.convert(l) for l in ls]

	def write(self, outfile):
		print(len(self.d), len(self.cnt))
		assert(len(self.d) == len(self.cnt))
		with open(outfile, 'w+') as f:
			items = [(v, k) for k, v in self.d.items()]
			items.sort()
			for v, k in items:
				f.write('{0} {1} {2}\n'.format(k, v, self.cnt[k]))

	# register tokens only appear in wv
	#   NOTE, only do counting on training set
	def register_words(self, wv, seq, count):
		for w in seq:
			if w in wv and w not in self.d:
				self.d[w] = len(self.d)
				self.cnt[w] = 0
			if w in self.cnt:
				self.cnt[w] = self.cnt[w] + 1 if count else self.cnt[w]

	#   NOTE, only do counting on training set
	def register_all_words(self, seq, count):
		for w in seq:
			if w not in self.d:
				self.d[w] = len(self.d)
				self.cnt[w] = 0
			if w in self.cnt:
				self.cnt[w] = self.cnt[w] + 1 if count else self.cnt[w]

def load_data(path):
	data = []
	with open(path, 'r') as f:
		for l in f:
			if l.strip() == '':
				continue
			toks = l.rstrip().split(' ')
			data.append(toks)
	return data


def get_glove_words(f):
	glove_words = set()
	for line in open(f, "r"):
		word = line.split()[0].strip()
		glove_words.add(word)
	return glove_words

def pad_ends(ls):
	return ['<bos>'] + ls + ['<eos>']

def pad(ls, length, symbol, pad_back = True):
	if len(ls) >= length:
		return ls[:length]
	if pad_back:
		return ls + [symbol] * (length -len(ls))
	else:
		return [symbol] * (length -len(ls)) + ls  


def make_vocab(args, glove_vocab, all_word_indexer, word_indexer, label_indexer, srcfile, labelfile, count):
	num_ex = 0
	for _, (src_orig, label_orig) in enumerate(zip(open(srcfile,'r'), open(labelfile, 'r'))):
		label_orig = label_orig.rstrip()
		src_orig = src_orig.rstrip()
		if args.lowercase == 1:
			src_orig = src_orig.lower()

		src = src_orig.split(' ')
		label = label_orig.split(' ')

		num_ex += 1

		all_word_indexer.register_all_words(src, count)
		word_indexer.register_words(glove_vocab, src, count)
		label_indexer.register_all_words(label, count)
	return num_ex


def convert(opt, word_indexer, all_word_indexer, label_indexer, source, label, output, num_ex):
	np.random.seed(opt.seed)
		
	max_seq_l = opt.max_seq_l + 2 #add 2 for BOS and EOS
	sources = np.zeros((num_ex, max_seq_l), dtype=int)
	all_sources = np.zeros((num_ex, max_seq_l), dtype=int)
	labels = np.zeros((num_ex, max_seq_l), dtype =int)
	source_lengths = np.zeros((num_ex,), dtype=int)
	ex_idx = np.zeros(num_ex, dtype=int)
	batch_keys = np.array([None for _ in range(num_ex)])
	
	ex_id = 0
	for _, (src_orig, label_orig) in enumerate(zip(open(source,'r'), open(label,'r'))):
		if src_orig.rstrip() == '':
			continue

		if opt.lowercase == 1:
			src_orig = src_orig.lower()

		src = pad_ends(src_orig.rstrip().split())
		label = pad_ends(label_orig.rstrip().split())

		src = pad(src, max_seq_l, '<blank>')
		src = word_indexer.convert_sequence(src)
		   
		label = pad(label, max_seq_l, '<bos>') # <bos> has idx 0!!!
		label = label_indexer.convert_sequence(label)

		all_src = pad(src, max_seq_l, '<blank>')
		all_src = all_word_indexer.convert_sequence(all_src)

		sources[ex_id] = np.array(src, dtype=int)
		all_sources[ex_id] = np.array(all_src, dtype=int)
		source_lengths[ex_id] = (sources[ex_id] != 0).sum() 
		labels[ex_id] = np.array(label, dtype=int)
		batch_keys[ex_id] = (source_lengths[ex_id])
		ex_id += 1
		if ex_id % 100000 == 0:
			print("{}/{} sentences processed".format(ex_id, num_ex))
	
	print(ex_id, num_ex)
	if opt.shuffle == 1:
		rand_idx = np.random.permutation(ex_id)
		sources = sources[rand_idx]
		all_sources = all_sources[rand_idx]
		source_lengths = source_lengths[rand_idx]
		labels = labels[rand_idx]
		batch_keys = batch_keys[rand_idx]
		ex_idx = rand_idx
	
	# break up batches based on source/target lengths
	sorted_keys = sorted([(i, p) for i, p in enumerate(batch_keys)], key=lambda x: x[1])
	sorted_idx = [i for i, _ in sorted_keys]
	# rearrange examples	
	sources = sources[sorted_idx]
	all_sources = all_sources[sorted_idx]
	labels = labels[sorted_idx]
	source_l = source_lengths[sorted_idx]
	ex_idx = rand_idx[sorted_idx]
	
	curr_l_src = 0
	batch_location = [] #idx where sent length changes
	for j,i in enumerate(sorted_idx):
		if batch_keys[i] != curr_l_src:
			curr_l_src = source_lengths[i]
			batch_location.append(j)
	if batch_location[-1] != len(sources): 
		batch_location.append(len(sources)-1)
	
	#get batch sizes
	curr_idx = 0
	batch_idx = [0]
	for i in range(len(batch_location)-1):
		end_location = batch_location[i+1]
		while curr_idx < end_location:
			curr_idx = min(curr_idx + opt.batch_size, end_location)
			batch_idx.append(curr_idx)

	batch_l = []
	source_l_new = []
	for i in range(len(batch_idx)):
		end = batch_idx[i+1] if i < len(batch_idx)-1 else len(sources)
		batch_l.append(end - batch_idx[i])
		source_l_new.append(source_l[batch_idx[i]])
		
		# sanity check
		for k in range(batch_idx[i], end):
			assert(source_l[k] == source_l_new[-1])
			assert(sources[k, source_l[k]:].sum() == 0)

	
	# Write output
	f = h5py.File(output, "w")	  
	f["source"] = sources
	f["label"] = labels
	f['all_source'] = all_sources
	f["source_l"] = np.array(source_l_new, dtype=int)
	f["batch_l"] = batch_l
	f["batch_idx"] = batch_idx
	f['ex_idx'] = ex_idx
	print("saved {} batches ".format(len(f["batch_l"])))
	f.close()  


def process(opt):
	all_word_indexer = Indexer()	# all tokens will be recorded
	word_indexer = Indexer()		# only glove tokens will be recorded
	label_indexer = Indexer(symbols=["<bos>", '<eos>'], num_oov=0)
	glove_vocab = get_glove_words(opt.glove)

	print("First pass through data to get vocab...")
	num_ex_train = make_vocab(opt, glove_vocab, all_word_indexer, word_indexer, label_indexer, opt.train, opt.train_label, count=True)
	print("Number of sentences in training: {0}, number of tokens: {1}/{2}".format(num_ex_train, len(word_indexer.d), len(all_word_indexer.d)))
	num_ex_valid = make_vocab(opt, glove_vocab, all_word_indexer, word_indexer, label_indexer, opt.valid, opt.valid_label, count=True)
	print("Number of sentences in valid: {0}, number of tokens: {1}/{2}".format(num_ex_valid, len(word_indexer.d), len(all_word_indexer.d)))
	num_ex_test = make_vocab(opt, glove_vocab, all_word_indexer, word_indexer, label_indexer, opt.test, opt.test_label, count=False)
	print("Number of sentences in test: {0}, number of tokens: {1}/{2}".format(num_ex_test, len(word_indexer.d), len(all_word_indexer.d)))

	print('Number of all tokens found: {0}'.format(len(all_word_indexer.d)))
	all_word_indexer.write(opt.output + '.allword.dict')
	
	print('Number of tokens collected: {0}'.format(len(word_indexer.d)))
	word_indexer.write(opt.output + ".word.dict")
	all_word_indexer.write(opt.output + ".allword.dict")
	label_indexer.write(opt.output + ".label.dict")
	print("vocab size: {}".format(len(word_indexer.d)))

	#assert(len(label_indexer.d) == 23+2)
	convert(opt, word_indexer, all_word_indexer, label_indexer, opt.train, opt.train_label, opt.output + "-train.hdf5", num_ex_train)
	convert(opt, word_indexer, all_word_indexer, label_indexer, opt.valid, opt.valid_label, opt.output + "-valid.hdf5", num_ex_valid)
	convert(opt, word_indexer, all_word_indexer, label_indexer, opt.test, opt.test_label, opt.output + "-test.hdf5", num_ex_test)


def main(opt):
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--dir', help="Path to the data dir", default = "data/ner/")
	parser.add_argument('--train', help="Path to training CONLL text ner data file.", default="train.source.txt")
	parser.add_argument('--train_label', help="Path to valid CONLL text ner label file.", default="train.label.txt")
	parser.add_argument('--valid', help="Path to training CONLL text ner data file.", default="valid.source.txt")
	parser.add_argument('--valid_label', help="Path to valid CONLL text ner label file.", default="valid.label.txt")
	parser.add_argument('--test', help="Path to test CONLL text ner data file.", default="test.source.txt")
	parser.add_argument('--test_label', help="Path to test CONLL text ner label file.", default="test.label.txt")
	parser.add_argument('--glove', help="Path to GloVe vectors", default="")
	parser.add_argument('--output', help="Prefix of the output file names.", type=str, default = "ner")
	parser.add_argument('--max_seq_l', help="The max sentence length", default=100, type=int)
	parser.add_argument('--batch_size', help="The max batch size", default=32, type=int)
	parser.add_argument('--seed', help="The random seed to shuffle data before batching", default=1, type=int)
	parser.add_argument('--lowercase', help="Whether to use lowercase for vocabulary.", type=int, default = 1)
	parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on source length).", type = int, default = 1)

	opt = parser.parse_args(opt)

	opt.train = opt.dir + opt.train
	opt.train_label = opt.dir + opt.train_label
	opt.valid = opt.dir + opt.valid
	opt.valid_label = opt.dir + opt.valid_label
	opt.test = opt.dir + opt.test
	opt.test_label = opt.dir + opt.test_label
	opt.output = opt.dir + opt.output

	process(opt)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
