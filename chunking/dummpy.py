import numpy as np

def aczel_alsina(x, alpha):
	p = np.power(-np.log(x), alpha)
	p = np.power(p.sum(), 1/alpha)
	return np.exp(-p)