import sys
import argparse

def load_conll(path):
	data = []
	cur_toks = []
	cur_pos = []
	cur_bio = []
	cnt = 0
	max_seq_l = 0
	with open(path, 'r+') as f:
		for l in f:
			l = l.strip()
			# the end of example
			if l == '':
				ls = []
				for tok, pos, bio in zip(cur_toks, cur_pos, cur_bio):
					ls.append((tok, pos, bio))
				data.append(ls)

				# clear cache
				cur_toks = []
				cur_pos = []
				cur_bio = []

				# stats
				cnt += 1
				max_seq_l = max(max_seq_l, len(ls))
			else:
				toks = l.split()
				if len(toks) != 3:
					print('must be 3 tokens: {0}'.format(l))
					assert(False)
				tok, pos, bio = toks
				cur_toks.append(tok)
				cur_pos.append(pos)
				cur_bio.append(bio)
	print('{0} examples loaded'.format(cnt))
	print('max seq length: {0}'.format(max_seq_l))
	

	source = [[p[0] for p in ex] for ex in data]
	pos = [[p[1] for p in ex] for ex in data]
	label = [[p[2] for p in ex] for ex in data]
	return source, pos, label


def write_to(ls, out_file):
	print('writing to {0}'.format(out_file))
	with open(out_file, 'w+') as f:
		for l in ls:
			f.write((l + '\n'))



parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', help="Path to the data dir", default="data/chunking/")
parser.add_argument('--data', help="Path to CONLL2000 chunking file", default="train.txt")
parser.add_argument('--output', help="Prefix to the path of output", default="train")


def main(args):
	opt = parser.parse_args(args)

	# append path
	opt.data = opt.dir + opt.data
	opt.output = opt.dir + opt.output

	source, pos, label = load_conll(opt.data)
	
	write_to([' '.join(ls) for ls in source], opt.output + '.source.txt')
	write_to([' '.join(ls) for ls in pos], opt.output + '.pos.txt')
	write_to([' '.join(ls) for ls in label], opt.output + '.label.txt')


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))