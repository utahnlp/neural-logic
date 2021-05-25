import torch
import pickle
import sys
import random
from model import OpeModuloModel, Model
import torchvision.datasets as datasets
from torchvision import transforms
import argparse
from utils import setseed, str2bool
from config_ijcai import *
from utils import load_mnist_dataset, load_eval_dataset
from consistency_evaluation import digit_sum_prod_and_consist
from associativity_evaluation import associativity
from commutativity_evaluation import commutativity
from distributivity_evaluation import distributivity
from operator_datasets_classes import data_dictio_by_labels


def load_digit_model(path_to_model):
    f_0 = Model()
    f_0.load_state_dict(torch.load(path_to_model))
    f_0.cuda()
    return f_0

def load_pair_model(path_to_model):
    f = OpeModuloModel()
    f.load_state_dict(torch.load(path_to_model))
    f.cuda()
    return f



if __name__=="__main__":


    parser= argparse.ArgumentParser()

    parser.add_argument('--test_baseline', type=str2bool, default=False, help='True if it the pipelined setting that its being tested. False if it the joint setting.')
    parser.add_argument('--seed_digit', type=int, default=20, help="Seed used to train the Digit model. (It should always be the same as the one used to train the operator models).")
    parser.add_argument('--seed_operator', type=int, default=20, help="Seed used to train the Sum and Prod operator models. (It should be the same as the one used to train the digit model).", required=True)
    parser.add_argument('--tnorm', type=str, default='prod', help='tnorm in use')
    parser.add_argument('--digit_size', type=int, default=5000, help="Size of the DIGIT set used.", required=True)
    parser.add_argument('--digit_lr', type=float, default=0.001, help='Learning rate used to train the Digit model used to label de NOISYPAIR data')
    parser.add_argument('--digit_optim', type=int, default=0, help='Optimizer used to train the Digit model used to label the NOISYPAIR data. 0 if SGD, 1 if Adam.')
    parser.add_argument('--digit_epochs', type=int, default=20, help='Number of epochs used to train the Digit classifier.')
    parser.add_argument('--ope_size', type=int, default=10000, help="Size of the NOISYPAIR set.", required=True)
    parser.add_argument('--data_option', type=int, default=2, help='data option used during training. It should always be 2.')
    parser.add_argument('--ope_lr', type=float, default=0.001, help='Learning rate used to train the Sum and Prod operator models.')
    parser.add_argument('--ope_optim', type=int, default=0, help='Optimizer used to train the Sum and Prod operator models. 0 if SGD, 1 if Adam.')
    parser.add_argument('--ope_epochs', type=int, default=20, help='Number of epochs used to train the Sum and Prod operator classifiers.')
    parser.add_argument('--joint_lr', type=float, default=0.001, help='Learning rate used for the joint learning process. (Only active if test_baseline is False)')
    parser.add_argument('--joint_optim', type=int, default=0, help='Optimizer used used for the joint learning process. (Only active if test_baseline is False). 0 if SGD, 1 if Adam.')    
    parser.add_argument('--lamb', type=float, default=0.001, help='Lambda coefficient used in the joint learning process (Only active if test_baseline is False).')
    parser.add_argument('--joint_epochs', type=int, default=20, help='Number of epochs used to train the joint classifier.')



    args = parser.parse_args()

    test_baseline = args.test_baseline
    seed_digit = args.seed_digit
    seed_operator = args.seed_operator
    tnorm = args.tnorm
    digit_size = args.digit_size
    digit_lr = args.digit_lr
    digit_optim = args.digit_optim
    digit_epochs = args.digit_epochs
    ope_size = args.ope_size
    option = args.data_option
    ope_lr = args.ope_lr
    ope_optim = args.ope_optim
    ope_epochs = args.ope_epochs
    joint_lr = args.joint_lr
    joint_optim = args.joint_optim
    lamb = args.lamb
    joint_epochs = args.joint_epochs

    #test_baseline = True
    #seed_digit = 10
    #seed_operator = 20
    #tnorm = 'prod'
    #digit_size = 5000
    #digit_lr = 0.001
    #digit_optim = 1
    #ope_size = 5000
    #option = 2
    #ope_lr = 0.005
    #ope_optim=0
    #joint_lr=0.01
    #joint_optim=0
    #lamb=0.1
    
    print('Calculating test acc. for model: ')
    print('test_baseline ', test_baseline)
    print('seed ', seed_digit)
    print('tnorm ', tnorm)
    print('digit_size ', digit_size)
    print('digit_lr ', digit_lr)
    print('digit_optim ', digit_optim)
    print('ope_size ', ope_size)
    print('option ', option)
    print('ope_lr ', ope_lr)
    print('ope_optim ', ope_optim)
    print('joint_lr ', joint_lr)
    print('joint_optim ', joint_optim)


    # we get the digit test set from the original MNIST that is invariant
    mnist_dataset_file = DATA_DIR + '/' + 'mnist_dataset_seed_20.pt'
    mnist_trainset, mnist_valset, mnist_testset = load_mnist_dataset(mnist_dataset_file)
    # dictionary used during the evaluation of assoc and dist on test sets that were created from mnist_testset
    MNIST_dict = data_dictio_by_labels(mnist_testset)
    
    # loading evaluation properties test data
    commut = load_eval_dataset(DATA_DIR + "/" + "commutativity_test_size_50000_seed_20.pt")
    assoc = load_eval_dataset(DATA_DIR + "/" +"associativity_test_size_50000_seed_20.pt")
    dist = load_eval_dataset(DATA_DIR + "/" + "distributivity_test_size_50000_seed_20.pt")

    #creating data loader to evaluate commut, assoc and dist, these set are use for all testing for every setting
    commut_dataloader = torch.utils.data.DataLoader(commut, batch_size=2048, shuffle=False,num_workers = 0) #batch size can 2048 made it smaller for now
    assoc_dataloader = torch.utils.data.DataLoader(assoc, batch_size=2048, shuffle=False,num_workers = 0)
    dist_dataloader = torch.utils.data.DataLoader(dist, batch_size=2048, shuffle=False,num_workers = 0)


    
    if test_baseline:
        # loading three models from baseline
        path_to_digit_model = MODEL_DIR + '/' + SINGLE_MODEL_NAME.format(digit_size, digit_lr, digit_optim, tnorm, seed_digit, digit_epochs)
        path_to_prod_model = MODEL_DIR + '/' + PRODUCT_MODEL_NAME.format(option, digit_size, ope_size, ope_lr, ope_optim, tnorm, seed_operator, ope_epochs)
        path_to_sum_model = MODEL_DIR + '/' + SUM_MODEL_NAME.format(option, digit_size, ope_size, ope_lr, ope_optim, tnorm, seed_operator, ope_epochs)

        digit_model = load_digit_model(path_to_digit_model)
        prod_model = load_pair_model(path_to_prod_model)
        sum_model = load_pair_model(path_to_sum_model)
    else:
        # loading joint three models
        path_to_joint_digit_model = JOINT_MODEL_DIR + "/" + JOINT_DIGIT_MODEL_NAME.format(option,digit_size,ope_size,joint_lr,joint_optim,tnorm,lamb,seed_digit,joint_epochs)
        path_to_joint_sum_model = JOINT_MODEL_DIR + "/" + JOINT_SUM_MODEL_NAME.format(option,digit_size,ope_size,joint_lr,joint_optim,tnorm,lamb,seed_operator,joint_epochs)
        path_to_joint_prod_model = JOINT_MODEL_DIR + "/" + JOINT_PROD_MODEL_NAME.format(option,digit_size,ope_size,joint_lr,joint_optim,tnorm,lamb,seed_operator,joint_epochs)

        joint_digit_model = load_digit_model(path_to_joint_digit_model)
        joint_prod_model = load_pair_model(path_to_joint_prod_model)
        joint_sum_model = load_pair_model(path_to_joint_sum_model)



    # the path remains the same for all testing as the test data set is the same for all datawa
    path_to_test_data = DATA_DIR + '/' + OPERATOR_MNIST_FILE.format(option, digit_size, ope_size, 50000, 20)
 
    # we are only loading the test set which is invariant across the different settings of pair (operator) data 
    _, _, operator_test = torch.load(path_to_test_data)

    # dataloader to test digit, sum, prod and consistency accuracies
    test_dataloader = torch.utils.data.DataLoader(operator_test, batch_size=1024, shuffle=False,num_workers = 0)
    
    #test_baseline = True

    if test_baseline:
        dm = digit_model
        sm = sum_model
        pm = prod_model

    else:
        dm = joint_digit_model
        sm = joint_sum_model
        pm = joint_prod_model



    print('Digit acc.', 'avg. Sum and Prod acc.', 'sum acc.', 'prod acc.', 'avg. Sum and Prod coherence acc.', 'Sum coherence acc.', 'Prod coherence acc.')

    print(digit_sum_prod_and_consist(dm, sm, pm, test_dataloader, digit_size=digit_size, ope_size=ope_size, opcion=option, tnorma=tnorm, baseline=test_baseline))
    print(' ')
    print('sum commut')
    print(commutativity(sm, commut_dataloader))
    print(' ')
    print('prod commut')
    print(commutativity(pm, commut_dataloader))
    print(' ')
    print('sum assoc')
    print(associativity(sm, assoc_dataloader,MNIST_dict))
    print('prod assoc')
    print(associativity(pm, assoc_dataloader,MNIST_dict))
    print(' ')
    ld, rd = distributivity(sm, pm, dist_dataloader, MNIST_dict)
    print('left dist')
    print(ld)
    print('right dist')
    print(rd)

