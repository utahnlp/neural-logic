import sys
#sys.path.insert(1, '/home/mattiamg/ring_reason')
import torch
from torch import nn
from model import OpeModuloModel, Model
import time
from operator_datasets_classes import operatorMNIST_1, operatorMNIST_2, operatorMNIST_3, testOpMNIST, Commut_MNIST, Assoc_MNIST, Dist_MNIST, data_dictio_by_labels
import matplotlib.pyplot as plt
import numpy as np
import random
import subprocess
from statistics import mean, stdev
from itertools import cycle
from config_ijcai import *
import math
import argparse
from utils import str2bool, setseed



def plotting(loss_im1, loss_im2, sum_loss, prod_loss, total_loss, plotname):

    x = list(range(len(loss_im1)))

    fig = plt.figure()
    plt.plot(x, loss_im1)
    plt.plot(x, loss_im2)
    plt.plot(x, sum_loss)
    plt.plot(x, prod_loss)
    plt.plot(x, total_loss)

    plt.legend(['Im1', 'Im2', 'Sum', 'Prod', 'Joint'], loc='upper right')
    plt.savefig('plots/{}.png'.format(plotname))
    plt.close(fig)



def delete_files(files):
    subprocess.call(['rm'] +  files)



def product_tnorm_loss(batch_probs_im1, batch_probs_im2, batch_probs_sum):
    # Implication
    return 1 - (batch_probs_im1 * batch_probs_im2) + ((batch_probs_im1 * batch_probs_im2) * batch_probs_sum)


def r_product_tnorm_loss(batch_probs_im1, batch_probs_im2, batch_probs_sum):
    # Implication
    denominator = batch_probs_im1 * batch_probs_im2

    numerator = batch_probs_sum

    quotient = numerator/(denominator+0.0001)
    
    all_ones_tensor = torch.ones(quotient.shape).cuda()

    out = torch.min(all_ones_tensor,quotient+0.0001)

    # Contrapostive
    numerator_contra = 1 - denominator
    denominator_contra =  1 - numerator
    quotient_contra = numerator_contra / (denominator_contra+0.0001)
    all_ones_tensor_contra = torch.ones(quotient_contra.shape).cuda()
    out_contra = torch.min(all_ones_tensor_contra,quotient_contra+0.0001)

    return out, out_contra


def luka_tnorm_loss(batch_probs_im1, batch_probs_im2, batch_probs_sum):
    """ There are 100 constraints per each element in the batch of the form f_0(Im1,Lb1)+f_0(Im2,Lb2) --> f_1(Sm, LbSum) st Lb1+Lb2=LbSum. Translating this to
     luka tnorm: min( 1 , 1 -  max( 0, (batch_probs_im1+batch_probs_im2) - 1) + batch_probs_sum ) for each one of the 100 constraints.
     In total, there are torch.Size([batch_size, 100]) outputs corresponding 100 constraints values per each element in the batch. """

    tensor_sum_in_max = batch_probs_im1+batch_probs_im2
    one_in_max = torch.ones_like(tensor_sum_in_max).cuda()
    right_part_of_max = tensor_sum_in_max-one_in_max
    zero_in_max = torch.zeros_like(right_part_of_max).cuda()
    maximum = torch.max(zero_in_max,right_part_of_max)
    onesminus = torch.ones_like(maximum).cuda()
    right_side_min = onesminus-maximum+batch_probs_sum
    onesmin = torch.ones_like(right_side_min).cuda()


    return torch.min(onesmin,right_side_min)



def godel_tnorm_loss(batch_probs_im1, batch_probs_im2, batch_probs_sum):

    conjunction = torch.min(batch_probs_im1,batch_probs_im2)
    all_ones_tensor = torch.ones(conjunction.shape).cuda()

    #print("1: ", 1 - conjunction)
    #print("2: ", batch_probs_sum)

    return torch.max(all_ones_tensor-conjunction, batch_probs_sum)


def training_joint(f_0=None, f_1=None, f_2=None,  optim=1, lr=0.001, n_epochs=100, digit_train_dataloader=None, pair_train_dataloader=None, val_dataloader=None, commut_dataloader=None, assoc_dataloader=None,
                 MNIST_dict=None, batch_size=10, weight=0.005, name=None, constraints=False, tnorm='prod', f_0_name=None, f_1_name=None, f_2_name=None,  plotname=None, test=False, option=1, godel_optim=0, godel_lr=0.01, godel_lambda=1 , seed=None, warm=-1):

    pretrained = False

    use_sigmoid = False

    one_constraint = False

    print('constraints', constraints)

    if constraints:
        const_indix = torch.tensor([(i, j, (i + j) % 10, (i * j) % 10) for i in range(10) for j in range(10)])
        print('cost_indix shape: ',const_indix.shape)
        
        if (torch.cuda.is_available()):
            const_indix=const_indix.cuda()


        """we get indices corresponding to the first element of the sum (first image), second element of the sum (second image) and the results (sum).
        we create a tensor that repeats this indices as many times as size of batch. This is because we have to use this indices for each entry in the batch"""
        indices_im_1 = const_indix[:,0].repeat(batch_size, 1)
        indices_im_2 = const_indix[:,1].repeat(batch_size, 1)
        indices_sumim = const_indix[:,2].repeat(batch_size, 1)
        indices_prodim = const_indix[:,3].repeat(batch_size,1)

    if pretrained:

        path_to_pretrained_f_0 = f_0_name
        path_to_pretrained_f_1 = f_1_name
        device = torch.device('cpu')
        
        # Initializing models
        f_0 = Model()   #MNIST Classifier
        f_1 = OpeModuloModel()   #SumMNIST Classifier
        
        # loading models
        f_0.load_state_dict(torch.load(path_to_pretrained_f_0))
        f_1.load_state_dict(torch.load(path_to_pretrained_f_1))
    

    if use_sigmoid:
        criterion = torch.nn.MultiLabelSoftMarginLoss(reduction='sum')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    # initializing type of optimizer
    if optim==1:
        optimizer = torch.optim.Adam(list(f_0.parameters()) + list(f_1.parameters()) + list(f_2.parameters()) , lr=lr)
    if optim==0:
        optimizer = torch.optim.SGD(list(f_0.parameters()) + list(f_1.parameters()) + list(f_2.parameters()), lr=lr)

    # Calling and naming softmax operator
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)

    # Warm restarts 
    # warm_param is the size of the pair train set divided by the batch size floor
    #warm_param = math.floor(len(pair_train_dataloader.dataset)/batch_size)
    #print('warm_param', warm_param+1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 200, T_mult=1)


    if (torch.cuda.is_available()):
        f_0.cuda()
        f_1.cuda()
        f_2.cuda()

    print('TNORM IN USE: ', tnorm)

    best_avg_ope = 0
    best_ep = 0

    num_updates = 0

    for epoch in range(n_epochs):
        
        # counters for epoch losses; used by the plotting function
        total_im1_loss = 0.0
        total_im2_loss = 0.0
        sumim_total_loss = 0.0
        sum_total_tnorm_loss = 0.0
        total_train_loss = 0
        
        # initializing counters for validation loss
        total_SumIm_val_loss = 0
        total_digit_val_loss = 0

        f_2.train()
        f_1.train()
        f_0.train()
        
        h = 0
        y = 0

        len_dl = len(pair_train_dataloader) 
        
        for itr, (batch_digit, batch_pair) in enumerate(zip(cycle(digit_train_dataloader), pair_train_dataloader)):
            optimizer.zero_grad()        
            
            # Calculating loss for digit classifier f_0
            # batch to train digit classifier f_0
            digit_image, digit_label = batch_digit
            
            if (torch.cuda.is_available()):
                digit_image = digit_image.cuda()
                digit_label = digit_label.cuda()
            
            # training
            current_batch_size = digit_image.shape[0]
            h += current_batch_size 
            
            pred = f_0(digit_image)
        
            if tnorm == 'prod' or tnorm == 'rprod':

                if use_sigmoid:
                    labels_one_hot = torch.nn.functional.one_hot(label, num_classes=10) #Shape (B, C)
                    loss = criterion(pred, labels_one_hot)
                else:
                    digit_loss = criterion(pred, digit_label)

            if tnorm == 'luka':

                if use_sigmoid:
                    image_probs = sigmoid(pred)
                else:
                    image_probs = softmax(pred)
                digit_loss = torch.sum(torch.gather(image_probs, 1, torch.unsqueeze(digit_label, -1)))

            
            if tnorm == 'godel' or tnorm == 'rgodel':

                image_probs = softmax(pred)
                correct_class_probs = torch.gather(image_probs, 1, torch.unsqueeze(digit_label, -1))
                
                if epoch < warm:
                    digit_loss = criterion(pred, digit_label)
                else:
                    digit_loss = torch.min(correct_class_probs)
                
            # Calculating loss for pair classifier f_1

            # batch to train pair classifier f_1
            img1_labeledpair, img2_labeledpair, cat_img, _, _ = batch_pair
            # OPERATOR IMAGES (WE DONT USE THE LABELS)
            img1 = img1_labeledpair[0]
            img2 = img2_labeledpair[0]

            if (torch.cuda.is_available()):
                img1 = img1.cuda()
                img2 = img2.cuda()
                cat_img = cat_img.cuda()

    
            current_batch_size = img1_labeledpair[0].shape[0]
            h += current_batch_size


            # logits for right and left images in the operation
            logits_img1 = f_0(img1)
            logits_img2 = f_0(img2)
            # logits for SUM  operation
            logits_sum_img = f_1(cat_img)

            # logits for PROD operator
            logits_prod_img = f_2(cat_img)
            # when using the constraints. this is an option. if constraints is False we bypass the constraints during learning (which is equivalent to have weight=0)
            if constraints:

                # output of the neural net passed through a Softmax layer, giving a prob dist  over the 10 possible labels
                
                # Probabilities for SUM
                sumim_probs = softmax(logits_sum_img)
                prodim_probs = softmax(logits_prod_img)
                # Probabilities for element of the operation
                img1_probs = softmax(logits_img1)
                img2_probs = softmax(logits_img2)
                
                # here we consider only one constraint per example. the constraint defined by the predictions of the SINGLE classifier
                if one_constraint:  
                    img1_labels = torch.argmax(img1_probs, dim=1, keepdim=True)
                    img2_labels = torch.argmax(img2_probs, dim=1, keepdim=True)
                    combined_label = (img1_labels + img2_labels)%10
                    
                    batch_probs_im1 = torch.gather(img1_probs, 1, img1_labels)
                    batch_probs_im2 = torch.gather(img2_probs, 1, img2_labels)
                    batch_probs_sum = torch.gather(sumim_probs, 1, combined_label)
                
                # else we consider all 100 constraints
                else:
                    # here we are handling the case of the final batch that may have diffent size than the rest so we create the indices just for those examples in the last batch
                    if indices_im_1.shape[0]!=img1_probs.shape[0]:
                        short_indices_im_1 = const_indix[:, 0].repeat(img1_probs.shape[0], 1)
                        short_indices_im_2 = const_indix[:, 1].repeat(img1_probs.shape[0], 1)
                        short_indices_sumim = const_indix[:, 2].repeat(img1_probs.shape[0], 1)
                        short_indices_prodim = const_indix[:, 3].repeat(img1_probs.shape[0], 1)

                        # we gather the indices of the constraints with their corresponent probabilities outputed by the models
                        batch_probs_im1 = torch.gather(img1_probs, 1, short_indices_im_1)
                        batch_probs_im2 = torch.gather(img2_probs, 1, short_indices_im_2)
                        batch_probs_sum = torch.gather(sumim_probs, 1, short_indices_sumim)
                        batch_probs_prod = torch.gather(prodim_probs, 1, short_indices_prodim)

                    else:
                        batch_probs_im1 = torch.gather(img1_probs, 1, indices_im_1)
                        batch_probs_im2 = torch.gather(img2_probs, 1, indices_im_2)
                        batch_probs_sum = torch.gather(sumim_probs, 1, indices_sumim)
                        batch_probs_prod = torch.gather(prodim_probs, 1, indices_prodim)

                # for each example we have to add over the neg log  probability of each one of the 100 constraints to get the contraint loss
                
                # if we are using Product t-norm we evaluate the probability of the contraints using the product t-norm connectors (function product_tnorm_loss)
                if tnorm == 'prod':
                    # (batch_size, 100)
                    product_constraint_batch_loss_sum = product_tnorm_loss(batch_probs_im1, batch_probs_im2, batch_probs_sum)
                    oneone = torch.ones(product_constraint_batch_loss_sum.shape).cuda()
                    # we add a small epsilon to make it work then we take the min to guarantee that it's below or equal to 1. 
                    product_constraint_batch_loss_sum = -torch.log(torch.min(oneone,product_constraint_batch_loss_sum+0.0001))
                    product_loss_sum = torch.sum(product_constraint_batch_loss_sum, dim=1) # shape([10])
                    product_loss_sum = torch.sum(product_loss_sum)
                    tnorm_loss_sum = product_loss_sum

                    product_constraint_batch_loss_prod = product_tnorm_loss(batch_probs_im1, batch_probs_im2, batch_probs_prod)
                    oneone = torch.ones(product_constraint_batch_loss_prod.shape).cuda()
                    product_constraint_batch_loss_prod = -torch.log(torch.min(oneone,product_constraint_batch_loss_prod+0.0001))
                    product_loss_prod = torch.sum(product_constraint_batch_loss_prod, dim=1) # shape([10])
                    product_loss_prod = torch.sum(product_loss_prod)
                    tnorm_loss_prod = product_loss_prod
    
                    total_loss = digit_loss+(weight*(tnorm_loss_sum+tnorm_loss_prod))

                # if we are using  R-Product t-norm we evaluate the probability of the contraints using the product t-norm connectors (function r_product_tnorm_loss)
                if tnorm == 'rprod':

                    rproduct_constraint_batch_loss_sum_impli, rproduct_constraint_batch_loss_sum_contra  = r_product_tnorm_loss(batch_probs_im1, batch_probs_im2, batch_probs_sum)
                    rproduct_constraint_batch_loss_impli = -torch.log(rproduct_constraint_batch_loss_sum_impli)
                    rproduct_constraint_batch_loss_contra = -torch.log(rproduct_constraint_batch_loss_sum_contra)
                    rproduct_loss_sum = torch.sum(rproduct_constraint_batch_loss_impli)
                    tnorm_loss_sum = rproduct_loss_sum

                    rproduct_constraint_batch_loss_prod, _  = r_product_tnorm_loss(batch_probs_im1, batch_probs_im2, batch_probs_prod)
                    rproduct_constraint_batch_loss_prod_log = -torch.log((rproduct_constraint_batch_loss_prod))
                    rproduct_loss_prod = torch.sum(rproduct_constraint_batch_loss_prod_log)
                    tnorm_loss_prod = rproduct_loss_prod
                    
                    total_loss = digit_loss+(weight*(tnorm_loss_sum+tnorm_loss_prod))

                # if we are using Product t-norm we evaluate the probability of the contraints using the product t-norm connectors (function luka_tnorm_loss)
                if tnorm == 'luka':

                    luka_constraint_batch_loss_sum = luka_tnorm_loss(batch_probs_im1, batch_probs_im2, batch_probs_sum)
                    luka_loss_sum = torch.sum(luka_constraint_batch_loss_sum)
                    tnorm_loss_sum = luka_loss_sum

                    luka_constraint_batch_loss_prod = luka_tnorm_loss(batch_probs_im1, batch_probs_im2, batch_probs_prod)
                    luka_loss_prod = torch.sum(luka_constraint_batch_loss_prod)
                    tnorm_loss_prod = luka_loss_prod

                    total_luka_sum = digit_loss+(weight*(tnorm_loss_sum+tnorm_loss_prod))
                    total_loss = -1 * total_luka_sum
                
                if tnorm == 'godel':

                    #if num_updates < 0:
                    if epoch < warm:

                        product_constraint_batch_loss_sum = product_tnorm_loss(batch_probs_im1, batch_probs_im2, batch_probs_sum)
                        oneone = torch.ones(product_constraint_batch_loss_sum.shape).cuda()
                        # we add a small epsilon to make it work then we take the min to guarantee that it's below or equal to 1. 
                        product_constraint_batch_loss_sum = -torch.log(torch.min(oneone,product_constraint_batch_loss_sum+0.0001))
                        product_loss_sum = torch.sum(product_constraint_batch_loss_sum, dim=1) # shape([10])
                        product_loss_sum = torch.sum(product_loss_sum)
                        tnorm_loss_sum = product_loss_sum

                        product_constraint_batch_loss_prod = product_tnorm_loss(batch_probs_im1, batch_probs_im2, batch_probs_prod)
                        oneone = torch.ones(product_constraint_batch_loss_prod.shape).cuda()
                        product_constraint_batch_loss_prod = -torch.log(torch.min(oneone,product_constraint_batch_loss_prod+0.0001))
                        product_loss_prod = torch.sum(product_constraint_batch_loss_prod, dim=1) # shape([10])
                        product_loss_prod = torch.sum(product_loss_prod)
                        tnorm_loss_prod = product_loss_prod

                        total_loss = digit_loss+(weight*(tnorm_loss_sum+tnorm_loss_prod))
                    
                    else: 
                        godel_constraint_batch_loss_sum = godel_tnorm_loss(batch_probs_im1, batch_probs_im2, batch_probs_sum)
                        godel_loss_sum = torch.min(godel_constraint_batch_loss_sum)
                        godel_constraint_batch_loss_prod = godel_tnorm_loss(batch_probs_im1, batch_probs_im2, batch_probs_prod)
                        godel_loss_prod = torch.min(godel_constraint_batch_loss_prod)
                        weight = godel_lambda
                        optim = godel_optim

                        # we are freezing the parameters of the Digit classifier f_0 after warm-up
                        if optim==1:
                            optimizer = torch.optim.Adam(list(f_1.parameters()) + list(f_2.parameters()) , lr=godel_lr)
                        if optim==0:
                            optimizer = torch.optim.SGD(list(f_1.parameters()) + list(f_2.parameters()), lr=godel_lr)
                        
                        total_loss = -1 * torch.min(digit_loss, weight*torch.min(godel_loss_sum, godel_loss_prod))
                         
            else:
                # if by-passing the tnorm regularizer
                total_loss = digit_loss
                
            total_train_loss += total_loss.item()
            
            # backward step and parameters update
            total_loss.backward()
            num_updates += 1
            optimizer.step()
            scheduler.step(epoch + itr / len_dl) 

        total_train_loss = total_train_loss / h 
        #print('total train loss each epoch ', total_train_loss)

        # VALIDATION
        
        f_2.eval()
        f_1.eval()
        f_0.eval()
        total_SumIm = 0
        total_ProdIm = 0
        total_singleIm = 0
        total_sum_consistency = 0
        total_prod_consistency = 0    
        
        for itr, (img1_labeledpair, img2_labeledpair, cat_img, label_sum, label_prod) in enumerate(val_dataloader):

            current_batch_size = img1_labeledpair[0].shape[0]
            y += current_batch_size
             

            # OPERATOR IMAGES AND LABELS
            img1 = img1_labeledpair[0]
            label_1 = img1_labeledpair[1]
            img2 = img2_labeledpair[0]
            label_2 = img2_labeledpair[1]

            if (torch.cuda.is_available()):

                img1 = img1.cuda()
                label_1 = label_1.cuda()
                img2 = img2.cuda()
                label_2 = label_2.cuda()
                cat_img = cat_img.cuda()
                label_sum = label_sum.cuda()
                label_prod = label_prod.cuda()

            # logits and probs for right and left images in the operation
            logits_img1 = f_0(img1)
            if use_sigmoid:
                img1_probs = torch.nn.functional.sigmoid(logits_img1)
            else:
                img1_probs = torch.nn.functional.softmax(logits_img1, dim=1)
            logits_img2 = f_0(img2)
            if use_sigmoid:
                img2_probs = torch.nn.functional.sigmoid(logits_img2)
            else:
                img2_probs = torch.nn.functional.softmax(logits_img2, dim=1)
            
            # logits and probs for SUM operation
            logits_sum_img = f_1(cat_img)
            
            if use_sigmoid:
                sumim_probs = torch.nn.functional.sigmoid(logits_sum_img)
            else:
                sumim_probs = torch.nn.functional.softmax(logits_sum_img, dim=1)
            
            logits_prod_img = f_2(cat_img)

            prodim_probs = torch.nn.functional.softmax(logits_prod_img, dim=1)

            # getting predicted labels from batch
            _, pred_labels_sumim = torch.max(sumim_probs, dim=1)
            _, pred_labels_prodim = torch.max(prodim_probs, dim=1)
            _, pred_labels_img1 = torch.max(img1_probs, dim=1)
            _, pred_labels_img2 = torch.max(img2_probs, dim=1)
           
    
            sum_eq = (pred_labels_sumim == label_sum)
            prod_eq = (pred_labels_prodim == label_prod)
            single_eq = (torch.cat((pred_labels_img1, pred_labels_img2), 0) == torch.cat((label_1, label_2), 0))
            
            total_SumIm += int(torch.sum(sum_eq.type(torch.FloatTensor))) 
            total_ProdIm += int(torch.sum(prod_eq.type(torch.FloatTensor)))
            total_singleIm += int(torch.sum(single_eq.type(torch.FloatTensor)))


            # CONSISTENCY
            # sum modulo 10 of two input images
            im1_plus_im2_mod_10 = (pred_labels_img1+pred_labels_img2)%10

            # calculating averaged difference between f_0 and f_1 to calculate consistency accuracy
            equals_consist_sum = (im1_plus_im2_mod_10 ==  pred_labels_sumim)
            total_sum_consistency += int(torch.sum(equals_consist_sum.type(torch.FloatTensor)).data.cpu())

            # prod modulo 10 of two input images
            im1_prod_im2_mod_10 = (pred_labels_img1*pred_labels_img2)%10

            equals_consist_prod = (im1_prod_im2_mod_10 ==  pred_labels_prodim)
            total_prod_consistency += int(torch.sum(equals_consist_prod.type(torch.FloatTensor)).data.cpu())



        # Total Accuracies for Sum Model (f_1), Single model (f_0) and Consistency
        accuracy_SumIm = total_SumIm / y
        accuracy_ProdIm = total_ProdIm / y

        # y is actually the size of the dev set
        # given that we are computing accuracy for single model using both im1 and im2 the size of the devset for single model is twice the size
        accuracy_singleIm = total_singleIm / (2*y)

        accuracy_sum_consistency = total_sum_consistency / y
        accuracy_prod_consistency = total_prod_consistency / y

        print('EPOCH ', epoch,' digit ',accuracy_singleIm, ' sum ', accuracy_SumIm, 'prod ', accuracy_ProdIm,' accuracy_sum_consistency ', accuracy_sum_consistency, 'acc_prod_cons', accuracy_prod_consistency)   
        # we strat saving the model at epoch 3: we observed that the accuracies are more stable from that point on
        if epoch>=3: 
            
            digit_train_size = len(digit_train_dataloader.dataset)
            pair_train_size = len(pair_train_dataloader.dataset)

            f_0_name = JOINT_MODEL_DIR + '/' + JOINT_DIGIT_MODEL_NAME.format(option, digit_train_size, pair_train_size, lr, optim, tnorm, weight, seed, n_epochs)
            f_1_name = JOINT_MODEL_DIR + '/' + JOINT_SUM_MODEL_NAME.format(option, digit_train_size, pair_train_size, lr, optim, tnorm, weight, seed, n_epochs)
            f_2_name = JOINT_MODEL_DIR + '/' + JOINT_PROD_MODEL_NAME.format(option, digit_train_size, pair_train_size, lr, optim, tnorm, weight, seed, n_epochs)
    
            # we are seleting the model from the epoch where the accuracy from Digit model was higher than 80% and with best average between Digit plus coherence constraints.
            # notice that this implies that we are usign only the signal from coherence and digit to select the best sum and produt models.
            if accuracy_singleIm > 0.80 and best_avg_ope < (accuracy_singleIm+accuracy_sum_consistency+accuracy_prod_consistency)/3:
                best_ep = epoch
                best_avg_ope = (accuracy_singleIm+accuracy_sum_consistency+accuracy_prod_consistency)/3
                if test!=True:
                    torch.save(f_0.state_dict(), f_0_name)
                    torch.save(f_1.state_dict(), f_1_name)
                    torch.save(f_2.state_dict(), f_2_name)



    print('Best epoch: ', best_ep)
    print('Best avg operator acc: ', best_avg_ope)
    return best_ep, best_avg_ope





if __name__ == "__main__":


    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    parser= argparse.ArgumentParser()

    parser.add_argument('--test', type=str2bool, default=False, help='True or False if it is a Test or not. If it is, models will not be saved.') 
    parser.add_argument('--DIGIT_size', type=int, default=5000, help='Size of the DIGIT dataset to train the Digit model.')
    parser.add_argument('--PAIR_size', type=int, default=5000, help='Size of the PAIR dataset used to train the Sum and Product models.')
    parser.add_argument('--data_option', type=int, default=2, help='Data option of the PAIR dataset for training.')
    parser.add_argument('--data_seed', type=int, default=20, help='Seed used to create the DIGIT and PAIR data sets we are training with (from create_DIGIT_PAIR.py).')
    parser.add_argument('--Dev_Test_size', type=int, default=50000, help='Size of the test and dev sets for PAIR.')
    parser.add_argument('--train_batch_size', type=int, default=64, help='batch size for training.')
    parser.add_argument('--validation_batch_size', type=int, default=1024, help='batch size for validation.')
    parser.add_argument('--tnorm', type=str, default='prod', help='t-norm to use. Note: If Godel is selected parameters learning_rate, optimizer and lambda_coef will be applied to the warming up process with prod tnorm. These parameters for Godel will be the inputs Godel_Optim, Godel_lambda and Godel_lr.')
    parser.add_argument('--nepochs', type=int, default=1600, help="Number of epochs to run.")
    parser.add_argument('--seed', type=int, default=20, help="Seed for the training de models.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--optimizer', type=int, default=0, help="optimizer, 0 for SGD, 1 for ADAM.")
    parser.add_argument('--lambda_coef', type=float, default=0.1, help='lambda coefficient for coherence(consistency) constraints')
    parser.add_argument('--warm_up_epochs', type=int, default=-1, help='Number of epochs to warm up Godel with s-prod t-norm. -1 means no warm up.')
    parser.add_argument('--Godel_Optim', type=int, default=0, help="Optimizer to use for Godel t-norm.")
    parser.add_argument('--Godel_lambda', type=float, default=0.01, help="Lambda coefficient for Godel t-norm.")
    parser.add_argument('--Godel_lr', type=float, default=0.01, help="Learning rate for Godel t-norm.")

    args = parser.parse_args()

    test_flag = args.test
    print('TEST?: ', test_flag)

    single_mnist_size = args.DIGIT_size
    operator_size = args.PAIR_size
    option = args.data_option
    data_seed = args.data_seed
    train_batch_size = args.train_batch_size
    tnorm_in_use = args.tnorm
    number_epochs=args.nepochs
    seed = args.seed
    warm = args.warm_up_epochs
    

    # for Godel t-norm
    godel_optim = args.Godel_Optim
    godel_lr = args.Godel_lr
    godel_lambda = args.Godel_lambda
        

    setseed(seed)
    single_mnist_file = "/uusoc/scratch/bluefish/mattiamg/ijcai2021/data/single_mnist_size_{}_seed_{}.pt".format(single_mnist_size,data_seed)
    
    digit_train, digit_val, digit_test, _  = torch.load(single_mnist_file)
    
    digit_dataloader = torch.utils.data.DataLoader(digit_train, batch_size=train_batch_size, shuffle=True, num_workers = 0)

    operator_mnist_file = "/uusoc/scratch/bluefish/mattiamg/ijcai2021/data/option_{}_mnist_size_{}_operator_size_{}_{}_seed_{}.pt".format(option, single_mnist_size, operator_size, args.Dev_Test_size, data_seed)
    pair_train, operator_val, operator_test = torch.load(operator_mnist_file)

    print('Datasets loaded...')


    pair_dataloader = torch.utils.data.DataLoader(pair_train, batch_size=train_batch_size, shuffle=True,num_workers = 0)

    val_dataloader = torch.utils.data.DataLoader(operator_val, batch_size=args.validation_batch_size, shuffle=False,num_workers = 0)

    MNIST_dict = data_dictio_by_labels(digit_val)

    for coef in [args.lambda_coef]:

        for optim in [args.optimizer]:    
            
            for learning_rate in [args.learning_rate]:

                
                if tnorm_in_use is not 'godel':
                    print('THIS MODEL:')
                    print('TNORM IN USE: ', tnorm_in_use)
                    print('SIZES: DIGIT {}, PAIR {} WITH OPTION {}'.format(single_mnist_size, operator_size, option))
                    print('BATCH SIZE TRAIN: ', train_batch_size)
                    print('LAMBDA: ', coef)
                    print('OPTIMIZER: ', optim)
                    print('LEARNING RATE: ', learning_rate)
                    print('TRAINING SEED ', seed)
                    print('NUM. EPOCHS', number_epochs)

                else:
                    print('GODEL TNORM IN USE')
                    print('TRAINING SEED ', seed)
                    print('BATCH SIZE TRAIN: ', train_batch_size)
                    print('NUM. EPOCHS', number_epochs)
                    print('WARM-UP MODEL WITH S-PRODUCT t-norm for ',warm, ' epochs.')
                    print('WARM-UP OPTIMIZER: ', optim)
                    print('WARM-UP LAMBDA: ', coef)
                    print('WARM-UP LEARNING RATE: ', learning_rate)
                    print('GODEL OPTIMIZER: ', godel_optim)
                    print('GODEL LEARNING RATE: ', godel_lr)
                    print('GODEL LAMBDA: ', godel_lambda) 
             

                setseed(seed) 
                f_0 = Model()
                setseed(seed)
                f_1 = OpeModuloModel()
                setseed(seed)
                f_2 = OpeModuloModel()
                training_joint(f_0=f_0, f_1=f_1, f_2=f_2,  optim=optim, lr=learning_rate, n_epochs=number_epochs, digit_train_dataloader=digit_dataloader, pair_train_dataloader=pair_dataloader, 
                        val_dataloader=val_dataloader, commut_dataloader=None, assoc_dataloader=None, MNIST_dict=None, batch_size=train_batch_size, weight=coef, name='test', constraints=True, tnorm=tnorm_in_use, 
                        f_0_name=None, f_1_name=None, f_2_name=None, plotname=None, test=test_flag, option=option, godel_optim=godel_optim, godel_lr=godel_lr, godel_lambda=godel_lambda, seed=seed, warm=warm)





