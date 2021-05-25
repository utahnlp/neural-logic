import torch
import pickle
import sys
import random
from model import OpeModuloModel, Model
import torchvision.datasets as datasets
from torchvision import transforms
import argparse
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



def digit_sum_prod_and_consist(digit_model, sum_model, prod_model, dataloader, digit_size=None, ope_size=None, opcion=None, tnorma=None, baseline=None):

    total_SumIm = 0
    total_ProdIm = 0
    total_singleIm = 0
    total_sum_consistency = 0
    total_prod_consistency = 0
    f_0 = digit_model.eval()
    f_1 = sum_model.eval()
    f_2 = prod_model.eval()
    y = 0
    mode = None

    if baseline is True:
        mode = 'BASELINE'
    else:
        mode = 'JOINT'

    #digit_model_output = open('/home/mattiamg/ring_reason/t_tests/{}_{}_DIGIT_digitsize{}_opesize{}_option{}.txt'.format(tnorma, mode, digit_size, ope_size, opcion), 'w')
    #sum_model_output = open('/home/mattiamg/ring_reason/t_tests/{}_{}_SUM_digitsize{}_opesize{}_option{}.txt'.format(tnorma, mode, digit_size, ope_size, opcion), 'w')
    #prod_model_output = open('/home/mattiamg/ring_reason/t_tests/{}_{}_PRODUCT_digitsize{}_opesize{}_option{}.txt'.format(tnorma, mode, digit_size, ope_size, opcion), 'w')
    

    #digit_model_ouput=open('{}BASEsum50050002.txt', 'w')
    for itr, (img1_labeledpair, img2_labeledpair, cat_img, label_sum, label_prod) in enumerate(dataloader):
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
        img1_probs = torch.nn.functional.softmax(logits_img1, dim=1)
        
        logits_img2 = f_0(img2)
        img2_probs = torch.nn.functional.softmax(logits_img2, dim=1)

        # logits and probs for SUM operation
        logits_sum_img = f_1(cat_img)
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

        #print(sum_eq.type(torch.FloatTensor))
        ''' code lines for the t-test files: uncomment to store them'''
        #for el in sum_eq.type(torch.FloatTensor):
            #print(el.item())
        #    sum_model_output.write(str(el.item()) + "\n")

        #for el in prod_eq.type(torch.FloatTensor):
        #    prod_model_output.write(str(el.item()) + "\n")

        #for el in single_eq.type(torch.FloatTensor):
        #    digit_model_output.write(str(el.item()) + "\n")

        
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

    '''closes t-test files'''
    #sum_model_output.close()
    #prod_model_output.close()
    #digit_model_output.close()

    # Total Accuracies for Sum Model (f_1), Single model (f_0) and Consistency
    accuracy_SumIm = total_SumIm / y
    accuracy_ProdIm = total_ProdIm / y

        # y is actually the size of the dev set
        # given that we are computing accuracy for single model using both im1 and im2 the size of the devset for single model is twice the size
    accuracy_singleIm = total_singleIm / (2*y)

    accuracy_sum_consistency = total_sum_consistency / y
    accuracy_prod_consistency = total_prod_consistency / y

    return accuracy_singleIm, (accuracy_SumIm+accuracy_ProdIm)/2, accuracy_SumIm, accuracy_ProdIm, (accuracy_sum_consistency+accuracy_prod_consistency)/2, accuracy_sum_consistency, accuracy_prod_consistency





