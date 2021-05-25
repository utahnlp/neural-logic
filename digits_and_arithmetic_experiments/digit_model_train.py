import sys
#sys.path.insert(1, '/home/mattiamg/ring_reason')
import torch
from torch import nn
from model import OpeModuloModel, Model
from associativity_evaluation import associativity
from commutativity_evaluation import commutativity
from config_ijcai import *
from mnist_training import training_mnist
import argparse
from utils import str2bool, setseed




# code to train the digit model
def digitmodel_train(size_digit_data=5000, data_seed=20, learning_rate=0.001, optimizer=1, train_batch_size=64,  n_epochs=100, seed=20, tnorm='prod', test=True):
    print('Training Digit Model with {} examples, lr {}, optimizer {}, batch size {},  n_epochs {}, tnorm {}, seed {}'.format(size_digit_data, learning_rate, optimizer, train_batch_size, n_epochs, tnorm, seed))

    single_mnist_file = DATA_DIR + '/' + SINGLE_MNIST_FILE.format(size_digit_data, data_seed)
    digit_train, digit_val, digit_test, _  = torch.load(single_mnist_file)
    digit_train_dataloader = torch.utils.data.DataLoader(digit_train, batch_size=train_batch_size, shuffle=True, num_workers = 0)
    digit_val_dataloader = torch.utils.data.DataLoader(digit_val, batch_size=512, shuffle=False, num_workers = 0)

    single_model_name = MODEL_DIR + '/' + SINGLE_MODEL_NAME.format(size_digit_data, learning_rate, optimizer, tnorm, seed, n_epochs)
    if test!=True:
        print('the model is stored at', single_model_name)
    setseed(seed)
    model_0 = Model()
    training_mnist(model=model_0, optim=optimizer, lr=learning_rate, n_epochs=n_epochs, train_dataloader=digit_train_dataloader, val_dataloader=digit_val_dataloader, lenmnistval=len(digit_val), name=single_model_name, tnorm=tnorm, test=test)





if __name__=="__main__":

    parser= argparse.ArgumentParser()

    parser.add_argument('--size_data', type=int, default=5000, help="Size of the DIGIT train set for Digit model.")
    parser.add_argument('--data_seed', type=int, default=20, help="seed used to create the DIGIT data")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--optimizer', type=int, default=0, help="optimizer, 0 SGD, 1 ADAM")
    parser.add_argument('--batch_size', type=int, default=64, help="train batch size")
    parser.add_argument('--nepochs', type=int, default=1600, help="Numer of epochs")
    parser.add_argument('--seed', type=int, default=20, help="seed to train")
    parser.add_argument('--tnorm', type=str, default='prod', help='t-norm to use')
    parser.add_argument('--test', type=str2bool, default=False, help='True or False if it is a Test or not. If it is, models will not be saved.')

    args = parser.parse_args()
    
    digitmodel_train(size_digit_data=args.size_data, data_seed=args.data_seed, learning_rate=args.learning_rate, optimizer=args.optimizer, train_batch_size=args.batch_size, n_epochs=args.nepochs, seed=args.seed, 
            tnorm=args.tnorm, test=args.test)






