#import sys
#sys.path.insert(1, '/home/mattiamg/ring_reason')
import torch
from torch import nn
from model import OpeModuloModel, Model
from utils import setseed, str2bool#, load_mnist_dataset, load_eval_dataset
import time
from operator_datasets_classes import operatorMNIST_1, operatorMNIST_2, operatorMNIST_3, testOpMNIST, Commut_MNIST, Assoc_MNIST, Dist_MNIST, data_dictio_by_labels
import numpy as np
import random
import subprocess
from statistics import mean, stdev
from mnist_training import training_mnist
from config_ijcai import *
import math
import argparse
from noisy_labeling import noisy_labeled_pairset


# this function trains the noisy labeled operator data without constraints. Notice that the noisy labeled data is like data augmentation with the contraints, before training. 
def training_operator(model=None, operator='sum',  optim=1, lr=0.001, n_epochs=100, ope_train_dataloader=None, ope_val_dataloader=None, name=None, tnorm=None, test=False, dataforloader=None, inner_lr=None, inner_optim=None, batch_size=64, warm=None):

    use_sigmoid=False

    saved_as = 'No model'

    if use_sigmoid:
        criterion = torch.nn.MultiLabelSoftMarginLoss(reduction='sum')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    if optim==1:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optim==0:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)

    warm_param = math.floor(len(ope_train_dataloader.dataset)/batch_size)
    print('warm_param', warm_param+1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,200, T_mult=1)


    if (torch.cuda.is_available()):
        model.cuda()

    no_epochs = n_epochs
    train_loss = list()
    val_loss = list()
    best_accuracy = 0.0
    best_ep = 0
    num_updates = 0
    for epoch in range(no_epochs):
        total_train_loss = 0
        total_val_loss = 0
        h = 0
        k = 0
        model.train()
        len_dl = len(ope_train_dataloader)

        for itr, (im1_labeledpair, im2_labeledpair, concat_image, label_sum, label_prod) in enumerate(ope_train_dataloader):

            current_batch_size = im1_labeledpair[0].shape[0]
            h += current_batch_size

            ope_image = concat_image

            if operator=='sum':
                ope_image_label = label_sum
            else:
                ope_image_label = label_prod

            if (torch.cuda.is_available()):
                ope_image = ope_image.cuda()
                ope_image_label = ope_image_label.cuda()

            optimizer.zero_grad()

            pred = model(ope_image)

            if tnorm == 'luka':
                if use_sigmoid:
                    image_probs = sigmoid(pred)
                else:
                    image_probs = softmax(pred)

                if epoch < warm:
                    loss = criterion(pred, ope_image_label)
                else:    
                    optim = inner_optim
                    lr = inner_lr

                    if optim==1:
                        optimizer = torch.optim.Adam(model.parameters(), lr=inner_lr)
                    if optim==0:
                        optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
                    loss = -torch.sum(torch.gather(image_probs, 1, torch.unsqueeze(ope_image_label, -1)))

            if tnorm == 'prod' or tnorm == 'rprod':
                if use_sigmoid:
                    labels_one_hot = torch.nn.functional.one_hot(ope_image_label,num_classes=10) #Shape (B, C)
                    loss = criterion(pred, labels_one_hot)
                else:
                    loss = criterion(pred, ope_image_label)

            
            if tnorm == 'godel' or tnorm == 'rgodel':
                
                image_probs = softmax(pred)
                correct_class_probs = torch.gather(image_probs, 1, torch.unsqueeze(ope_image_label, -1))
                
                if epoch < warm: 
                    loss = criterion(pred, ope_image_label)
                else:
                    optim = inner_optim
                    lr = inner_lr

                    if optim==1:
                        optimizer = torch.optim.Adam(model.parameters(), lr=inner_lr)
                    if optim==0:
                        optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
                
                    loss = -1 * torch.min(correct_class_probs)

            total_train_loss += loss.item()

            loss.backward()
            num_updates += 1
            optimizer.step()
            scheduler.step(epoch + itr / len_dl)


        total_train_loss = total_train_loss / h
        train_loss.append(total_train_loss)


        # validation
        model.eval()
        total = 0
        counts_dict = {}

        for itr, (im1_labeledpair, im2_labeledpair, concat_image, label_sum, label_prod) in enumerate(ope_val_dataloader):

            current_batch_size_eval = im1_labeledpair[0].shape[0]
            k += current_batch_size_eval

            ope_image = concat_image

            if operator=='sum':
                ope_image_label = label_sum

            else:
                ope_image_label = label_prod


            if (torch.cuda.is_available()):
                ope_image = ope_image.cuda()
                ope_image_label = ope_image_label.cuda()

            pred = model(ope_image)

            if use_sigmoid:
                    labels_one_hot = torch.nn.functional.one_hot(ope_image_label,num_classes=10) #Shape (B, C)
                    loss = criterion(pred, labels_one_hot)
            else:
                    loss = criterion(pred, ope_image_label)

            total_val_loss += loss.item()

            if use_sigmoid:
                pred = torch.nn.functional.sigmoid(pred)
            else:
                pred = torch.nn.functional.softmax(pred, dim=1)

            _, pred_labels_opeim = torch.max(pred, dim=1)
            ope_eq = (pred_labels_opeim == ope_image_label)
            total += int(torch.sum(ope_eq.type(torch.FloatTensor)))

        accuracy = total / k

        total_val_loss = total_val_loss / k
        val_loss.append(total_val_loss)
        print('\nEpoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch + 1, no_epochs, total_train_loss, total_val_loss, accuracy))
        

        if best_accuracy < accuracy:
            best_ep = epoch+1
            best_accuracy = accuracy
            saved_as = name
            if test!=True:
                torch.save(model.state_dict(), saved_as)

    print('best accuracy', best_accuracy, 'best epoch', best_ep)
    return best_accuracy, saved_as



# this main function gets all the information need to find the righ data and models to perform the training with the augmented operator data
def train_with_noisypair_data(option=1, pair_size=5000, digit_size=5000, val_test_size=50000, data_seed=20, learning_rate=0.001, optimizer=1, tnorm='prod', digit_model_seed=20, epochs=100, operator='sum', optim_for_train=0, lr_for_train=0.001, train_batch_size=64, valid_batch_size=64, n_epochs_for_train=100, warm=-1, inner_lr=1, inner_optim=0, seed_train=None, test=True):
   
    path_to_pair_unlabeled_data = DATA_DIR + '/' + OPERATOR_MNIST_FILE.format(option, digit_size, pair_size, val_test_size, data_seed)
    labeled_train_data, operator_val, operator_test = torch.load(path_to_pair_unlabeled_data)
    noisy_pair_labeled_file = DATA_DIR +'/'+NOISY_LABELED_PAIR.format(option,pair_size,data_seed,digit_size,learning_rate, optimizer, tnorm, digit_model_seed, epochs)
    noisy_pair_labeled = torch.load(noisy_pair_labeled_file)
    if operator == 'sum':
        path_for_model = MODEL_DIR + '/' +  SUM_MODEL_NAME.format(option, digit_size, pair_size, lr_for_train, optim_for_train, tnorm, seed_train, n_epochs_for_train) 
    else:
        path_for_model = MODEL_DIR + '/' +  PRODUCT_MODEL_NAME.format(option, digit_size, pair_size, lr_for_train, optim_for_train, tnorm, seed_train, n_epochs_for_train) 
    setseed(seed_train)
    f_1 = OpeModuloModel()
    train_noisy_dataloader = torch.utils.data.DataLoader(noisy_pair_labeled, batch_size=train_batch_size, shuffle=True, num_workers = 0)
    develop_dataloader = torch.utils.data.DataLoader(operator_val, batch_size=valid_batch_size, shuffle=False, num_workers = 0)
    training_operator(model=f_1, operator=operator, optim=optim_for_train, lr=lr_for_train, n_epochs=n_epochs_for_train, ope_train_dataloader=train_noisy_dataloader, ope_val_dataloader=develop_dataloader, name=path_for_model, tnorm=tnorm, test=test, dataforloader=labeled_train_data, inner_lr=inner_lr, inner_optim=inner_optim, batch_size=train_batch_size, warm=warm)



if __name__=="__main__":




    parser= argparse.ArgumentParser()

    parser.add_argument('--test', type=str2bool, default=False, help='True or False if it is a Test or not. If it is, models will not be saved.')
    parser.add_argument('--data_seed', type=int, default=20, help="Seed used to create the PAIR and NOISYPAIR data")
    parser.add_argument('--DIGIT_size', type=int, default=10000, help="Size of the DIGIT data used to train the Digit classifier used to label the NOISYPAIR data.", required=True)
    parser.add_argument('--PAIR_size', type=int, default=10000, help="Size of the NOISYPAIR set.", required=True)
    parser.add_argument('--data_option', type=int, default=2, help='data option of the NOISYPAIR data')
    parser.add_argument('--PAIR_val_test_size', type=int, default=50000, help="Size of the PairDEV and PairTEST sets.")
    parser.add_argument('--DigitModel_lr', type=float, default=0.001, help='Learning rate used to train the Digit model used to label de NOISYPAIR data')
    parser.add_argument('--Digit_Optimizer', type=int, default=0, help='Optimizer used to train the Digit model used to label the NOISYPAIR data. 0 if SGD, 1 if Adam.')
    parser.add_argument('--Digit_nepochs', type=int, default=100, help='Number of epochs used to train the Digit model')
    parser.add_argument('--Digit_seed', type=int, default=20, help='seed used to train the Digit model')
    parser.add_argument('--tnorm', type=str, default='prod', help='tnorm in use') 
    parser.add_argument('--arithmethic_operator', type=str, default='sum', help='what operator model is being trained Sum (sum) or Product (prod)')
    parser.add_argument('--train_batch_size', type=int, help='training batch size.', default=64)
    parser.add_argument('--valid_batch_size', type=int, default=1024, help='validation batch size.')
    parser.add_argument('--training_n_epochs', default=100, help='number of epochs for training', type=int)
    parser.add_argument('--training_optimizer', type=int, default=0, help='Optimizer to train the arithmetic operator model. 0 is SGD, 1 if Adam.')
    parser.add_argument('--training_lr', type=float, default=0.01, help='Training learning rate')
    parser.add_argument('--training_seed', type=int, default=20, help='Seed used to train de arithmetic model')
    parser.add_argument('--warm_nepochs', type=int, default=-1, help='number of epoch warm-up')
    parser.add_argument('--warm_optim', type=int, default=0, help='optmizer for warm-up; 0 is SGD, 1 if Adam')
    parser.add_argument('--warm_lr', type=float, default=0,  help='learning rate for warm-up')
    
    args = parser.parse_args()

    test_flag = args.test
    option = args.data_option
    digit_size = args.DIGIT_size
    pair_size = args.PAIR_size
    optimi = args.training_optimizer
    learning_r = args.training_lr
    tnorm = args.tnorm
    operator = args.arithmethic_operator
    t_batch_s = args.train_batch_size
    v_batch_s = args.valid_batch_size
    n_epochs = args.training_n_epochs
    warm = args.warm_nepochs
    inner_optim = args.warm_optim
    inner_lr = args.warm_lr
    seed_train = args.training_seed


    print('Training ', operator, ' operator')
    print('test_flag', test_flag)
    print('DIGIT size: ', digit_size)
    print('(NOISY)PAIR size: ', pair_size)
    print('option: ', option)
    print('optimizer:', optimi)
    print('lr: ', learning_r)
    print('tnorm: ', tnorm)
    print('operator: ', operator)
    print('batch size: ', t_batch_s)
    print('seed: ', seed_train)
    if warm is not -1:
        print('numer of warm-up epochs: ', warm)
        print('warm_optim: ', inner_optim)
        print('inner_lr: ', inner_lr)

    
    
    train_with_noisypair_data(option=option, pair_size=pair_size, digit_size=digit_size, val_test_size=args.PAIR_val_test_size, data_seed=args.data_seed, learning_rate=args.DigitModel_lr, optimizer=args.Digit_Optimizer, tnorm=tnorm, digit_model_seed=args.Digit_seed, epochs=args.Digit_nepochs, operator=operator, optim_for_train=optimi, lr_for_train=learning_r, train_batch_size=t_batch_s, valid_batch_size=v_batch_s,  n_epochs_for_train=n_epochs, warm=warm, inner_lr=inner_lr, inner_optim=inner_optim, seed_train=seed_train, test=test_flag)
    

