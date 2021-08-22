import sys
import argparse

def load_conll(path):
	data = []
	cur_toks = []
	cur_pos = []
	cur_chunk = []
	cur_ner = []
	cnt = 0
	max_seq_l = 0
	with open(path, 'r+') as f:
		for l in f:
			l = l.strip()
			# the end of example
			if l == '':
				ls = []
				for tok, pos, chunk, ner in zip(cur_toks, cur_pos, cur_chunk, cur_ner):
					ls.append((tok, pos, chunk, ner))
				data.append(ls)

				# clear cache
				cur_toks = []
				cur_pos = []
				cur_chunk = []
				cur_ner = []

				# stats
				cnt += 1
				max_seq_l = max(max_seq_l, len(ls))
			else:
				toks = l.split()
				if len(toks) != 4:
					print('must be 4 tokens: {0}'.format(l))
					assert(False)
				tok, pos, chunk, ner = toks
				cur_toks.append(tok)
				cur_pos.append(pos)
				cur_chunk.append(chunk)
				cur_ner.append(ner)
	print('{0} examples loaded'.format(cnt))
	print('max seq length: {0}'.format(max_seq_l))
	

	source = [[p[0] for p in ex] for ex in data]
	pos = [[p[1] for p in ex] for ex in data]
	chunk = [[p[2] for p in ex] for ex in data]
	label = [[p[3] for p in ex] for ex in data]
	return source, pos, chunk, label


def write_to(ls, out_file):
	print('writing to {0}'.format(out_file))
	with open(out_file, 'w+') as f:
		for l in ls:
			f.write((l + '\n'))



parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', help="Path to the data dir", default="data/ner/")
parser.add_argument('--data', help="Path to CONLL2003 ner file", default="train.txt")
parser.add_argument('--output', help="Prefix to the path of output", default="train")


def main(args):
	opt = parser.parse_args(args)

	# append path
	opt.data = opt.dir + opt.data
	opt.output = opt.dir + opt.output

	source, pos, chunk, label = load_conll(opt.data)
	
	write_to([' '.join(ls) for ls in source], opt.output + '.source.txt')
	write_to([' '.join(ls) for ls in pos], opt.output + '.pos.txt')
	write_to([' '.join(ls) for ls in chunk], opt.output + '.chunk.txt')
	write_to([' '.join(ls) for ls in label], opt.output + '.label.txt')


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))