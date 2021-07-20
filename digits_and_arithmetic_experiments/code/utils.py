import random
import torch
import numpy as np




def setseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')



def load_mnist_dataset(filename):

    print('Loading dataset from file ', filename)

    mnist_trainset, mnist_valset, mnist_testset = torch.load(filename)

    return mnist_trainset, mnist_valset, mnist_testset


def load_eval_dataset(filename):

    print('Loading dataset from file ', filename)

    dataset = torch.load(filename)

    return dataset
