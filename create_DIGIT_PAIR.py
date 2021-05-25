import sys
#sys.path.insert(1, '/home/mattiamg/ring_reason')
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import numpy as np
import random
import argparse
from operator_datasets_classes import operatorMNIST_1, operatorMNIST_2, operatorMNIST_3, testOpMNIST
#from config import *
from config_ijcai import *


def setseed(seed=20):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# function to save datasets: train, val, test under a given path)
def save_datasets(mnist_trainset, mnist_valset, mnist_testset, filename):

    torch.save((mnist_trainset,mnist_valset, mnist_testset), filename)
    print('Saved data file with name : ', filename)


def calculate_operator_dataset_sum(dataset):
    """
    Find the sum of all images in the dataset
    """

    all_images = []
    for i in range(len(dataset)):
        _, _, img, _, _ = dataset[i]
        all_images.append(img)
    return torch.sum(torch.cat(all_images))


"""
 This function gets the MNIST data as input that has previously been split into train, dev and test using a certain seed by the program create_MNIST_train_dev_test.py.
 
 It gets the desired size for the (to be created) DIGIT set as a parameter: digit_train_size. It (randomly using the given input seed) selects digit_train_size examples from the input train set and selects the same number of examples from the remaining set of examples that were not selected from train set. 
 These two sets are the DIGIT set (that is used to train the Digit classifier), and the set of examples from which the PAIR data is created.
 
 It returns new_trainset which is the DIGIT set, not_used_train_set which is the set used to create PAIR later, and mnist_valset and mnist_testset which are exactly the same that were given as input. 
"""

def sample_digitdata_and_notused(mnist_trainset, mnist_valset, mnist_testset, digit_train_size=10000, seed=20):

    setseed(seed=seed) # Sets seed to 20

    if len(mnist_trainset) != digit_train_size:
        new_trainset, not_used_mnistset =  torch.utils.data.random_split(mnist_trainset, [digit_train_size, len(mnist_trainset) - digit_train_size])
        

    if len(not_used_mnistset) < digit_train_size:
        print('There are not enough example to build the not used data set STOP!')
    
    not_used_train_set, _  = torch.utils.data.random_split(not_used_mnistset, [digit_train_size, len(not_used_mnistset) - digit_train_size])


    print('Size of original mnist set : ', len(mnist_trainset))
    print('Size of created training set : ', len(new_trainset))
    print('Size of validation set : ', len(mnist_valset))
    print('Size of test set : ', len(mnist_testset))
    print('Size of not used set : ', len(not_used_train_set))
    # not_used_train_set and mnist_trainset are disjoint. mnist_trainset is used to train the DIGIT model and not_used_train_set is used to create the PAIR data later.

    return new_trainset, mnist_valset, mnist_testset, not_used_train_set

# helper function to load data 
def load_mnist_dataset(filename):

    mnist_trainset, mnist_valset, mnist_testset = torch.load(filename)

    return mnist_trainset, mnist_valset, mnist_testset


if __name__=="__main__":

    parser= argparse.ArgumentParser()

    parser.add_argument('--mnist_seed', type=int, default=20, help="Seed that was used in create_MNIST_train_dev_test.py to create the MNIST split into train, validation, test.")
    parser.add_argument('--single_size', type=int, default=10000, help="Size of the DIGIT set (to train single (Digit) mnist model).")
    parser.add_argument('--operator_train_size', type=int, default=10000, help="Size of the train set for operator mnist models.")
    parser.add_argument('--operator_test_size', type=int, default=50000, help="Size of the val/test set for operator mnist models.")
    parser.add_argument('--option', type=int, default=1, help="Dataset setting/option for training operator models. Can be one of (1,2,3)")
    parser.add_argument('--seed', type=int, default=20, help="Sets seed")

    args = parser.parse_args()

    # reading MNIST train,validation and test sets (TRAIN, DEV and TEST in the paper)
    mnist_dataset_file = DATA_DIR + '/' + 'mnist_dataset_seed_{}.pt'.format(args.mnist_seed)
    print(mnist_dataset_file)

    # loading MNIST train,validation and test sets (TRAIN, DEV and TEST in the paper).
    mnist_trainset, mnist_valset, mnist_testset = load_mnist_dataset(mnist_dataset_file)

    # creating the name of the file where the DIGIT data will be stored
    single_mnist_dataset_file = DATA_DIR + '/' + SINGLE_MNIST_FILE.format(args.single_size, args.seed) 

    # creating DIGIT dataset with the given size together with the (same as the input) validation set, test set, and the not_used set that will be use to create Operator dataset (PAIR in the paper)
    mnist_trainset, mnist_valset, mnist_testset, not_used_trainset = sample_digitdata_and_notused(mnist_trainset, mnist_valset, mnist_testset, digit_train_size=args.single_size, seed=args.seed)

    # saving DIGIT along with train, dev and not_used under the name created above
    torch.save((mnist_trainset,mnist_valset, mnist_testset, not_used_trainset), single_mnist_dataset_file)

    # creating the name of the file where the OPERATOR (called PAIR in the paper)  data will be stored
    operator_dataset_file = DATA_DIR + '/' + OPERATOR_MNIST_FILE.format(args.option, args.single_size,
                                                                        args.operator_train_size,
                                                                        args.operator_test_size,
                                                                        args.seed)
    # setting seed
    setseed(seed=args.seed)

    # given the type of PAIR data (Operator data) we create it accordingly
    """NOTE: In the paper we only use option 2"""
    if args.option == 1:
        operator_train = operatorMNIST_1(not_used_trainset, size=args.operator_train_size)
    elif args.option == 2:
        operator_train = operatorMNIST_2(not_used_trainset, size=args.operator_train_size)
    elif args.option == 3:
        operator_train = operatorMNIST_3(not_used_trainset, size=args.operator_train_size)
    else:
        raise NotImplementedError('Option size should be in (1,2,3), provided: ', args.option)

    # print the sum of the images to check that the seed is working
    print('Sum of operator train all images : ', calculate_operator_dataset_sum(operator_train))
    
    # setting seed
    setseed(seed=args.seed)

    # creating validation operator set (called PairDEV in the paper)
    operator_val = testOpMNIST(mnist_valset, size=args.operator_test_size)
    print('Sum of operator val all images : ', calculate_operator_dataset_sum(operator_val))

    setseed(seed=args.seed)
    #creating test operator set (called PairTEST in the paper)
    operator_test = testOpMNIST(mnist_testset, size=args.operator_test_size)
    print('Sum of operator test all images : ', calculate_operator_dataset_sum(operator_test))

    # saving the PAIR (operator) set along with the PairDEV (validation)  set and PairTEST (test set) 
    """NOTE: the validation and test sets for the operator set are invariant across sizes of the data: --single_size, --operator_test_size.
    If the same seed and same --operator_test_size are used the operator_val, operator_test are always the same"""
    save_datasets(operator_train, operator_val, operator_test, operator_dataset_file)


