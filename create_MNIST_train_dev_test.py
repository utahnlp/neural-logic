import sys
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import numpy as np
import random
import argparse
from config_ijcai import *

def setseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



def save_datasets(mnist_trainset, mnist_valset, mnist_testset, filename):

    torch.save((mnist_trainset,mnist_valset, mnist_testset), filename)
    print('Saved data file with name : ', filename)



def sample_train_dev(seed=20):
    """
    Sample 50k train samples, 10k validation samples, and 10k test samples
    Then save them in a folder. These are the sets respectively called TRAIN, DEV and TEST in the paper.
    """
    setseed(seed)

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True,
                                        transform=transforms.Compose([transforms.ToTensor()]))

    mnist_trainset, mnist_valset = torch.utils.data.random_split(mnist_trainset, [50000, 10000])

    mnist_testset = datasets.MNIST(root='./data', train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor()]))

    save_datasets(mnist_trainset, mnist_valset, mnist_testset, DATA_DIR + '/' + 'mnist_dataset_seed_{}.pt'.format(seed))


if __name__=="__main__":


    parser= argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20, help="Seed to Sample 50k train samples, 10k validation samples, and 10k test samples then save them in a folder")
    args = parser.parse_args() 
    sample_train_dev(seed=args.seed)


