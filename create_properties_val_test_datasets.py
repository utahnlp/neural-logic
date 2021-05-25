import sys
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import numpy as np
import random
import argparse
from operator_datasets_classes import Assoc_MNIST, Commut_MNIST, Dist_MNIST
from config_ijcai import *

def setseed(seed=20):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# helper function to load data
def load_mnist_dataset(filename):

    mnist_trainset, mnist_valset, mnist_testset = torch.load(filename)

    return mnist_trainset, mnist_valset, mnist_testset

# helper function to save the data
def save_eval_data(dataset, filename):

    torch.save(dataset, filename)
    print('Saved data file with name : ', filename)

"""This program calls the functions to create validation and test sets for the ring properties: Commutativity, Associativity and Distributivity with the given inputs
and stores them."""

if __name__=="__main__":


    parser= argparse.ArgumentParser()

    
    parser.add_argument('--mnist_seed', type=int, default=20, help='seed that was used to create the initial MNIST TRAIN, DEV, TEST split (in create_MNIST_train_dev_test.py')
    parser.add_argument('--associativity_size', type=int, default=50000, help="Size of the associativity set .")
    parser.add_argument('--commutativity_size', type=int, default=50000, help="Size of the commutativity set .")
    parser.add_argument('--distributivity_size', type=int, default=50000, help="Size of the distributivity set .")
    parser.add_argument('--seed', type=int, default=20, help="Sets seed")

    args = parser.parse_args()


    mnist_dataset_file =  DATA_DIR + '/' + 'mnist_dataset_seed_{}.pt'.format(args.mnist_seed)
    mnist_trainset, mnist_valset, mnist_testset = load_mnist_dataset(mnist_dataset_file)
    

    print('Creating the val datasets ... ')
    setseed(args.seed)
    assoc_val = Assoc_MNIST(mnist_valset, size=args.associativity_size)
    setseed(args.seed)
    comm_val = Commut_MNIST(mnist_valset, size=args.commutativity_size)
    setseed(args.seed)
    distri_val = Dist_MNIST(mnist_valset, size=args.distributivity_size)

    setseed(args.seed)

    print('Creating the test datasets ... ')
    assoc_test = Assoc_MNIST(mnist_testset, size=args.associativity_size)
    setseed(args.seed)
    comm_test = Commut_MNIST(mnist_testset, size=args.commutativity_size)
    setseed(args.seed)
    distri_test = Dist_MNIST(mnist_testset, size=args.distributivity_size)

    save_eval_data(assoc_val, DATA_DIR + '/' + ASSOCIATIVITY_MNIST_VAL.format(args.associativity_size, args.seed))
    save_eval_data(comm_val, DATA_DIR + '/' + COMMUTATIVITY_MNIST_VAL.format(args.commutativity_size, args.seed))
    save_eval_data(distri_val, DATA_DIR + '/' + DISTRIBUTIVITY_MNIST_VAL.format(args.distributivity_size, args.seed))

    save_eval_data(assoc_test, DATA_DIR + '/' + ASSOCIATIVITY_MNIST_TEST.format(args.associativity_size, args.seed))
    save_eval_data(comm_test, DATA_DIR + '/' + COMMUTATIVITY_MNIST_TEST.format(args.commutativity_size, args.seed))
    save_eval_data(distri_test, DATA_DIR + '/' + DISTRIBUTIVITY_MNIST_TEST.format(args.distributivity_size, args.seed))
