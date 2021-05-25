import torch
import pickle
import sys
import random
from model import OpeModuloModel, Model
import torchvision.datasets as datasets
from torchvision import transforms
import argparse


def images_from_pred_labels(pred_labels, test_images_MNIST):
    
    im_pred_labels = torch.empty(pred_labels.shape[0],1,28,28)
    if (torch.cuda.is_available()):
        im_pred_labels = im_pred_labels.cuda()

    for i in range(pred_labels.shape[0]):
        label = int(pred_labels[i])
        mapita = test_images_MNIST[label]
        image = random.choice(mapita)
        image = image[0].cuda()
        im_pred_labels[i]=image

    return im_pred_labels


def majority_labels(cato):
    # cato.shape = (batch, k_sample)

    final = torch.empty([cato.shape[0]])
    i=0
    for row in cato:

        rc = torch.stack([(row==x_u).sum() for x_u in list(range(10))])
        _,index = torch.max(rc, dim=0)
        final[i]=index
        i+=1
    return final



def distributivity(sum_model, prod_model, dataloader, dev_images_MNIST):

    sum_model.eval()
    prod_model.eval()

    total_left_distributivity = 0
    total_right_distributivity = 0
    size_of_data = 0
    k_sample = 5

    for itr, (image_bplusc, image_atimesb, image_atimesc, image_btimesa, image_ctimesa,
                               image_1, image_2, image_3) in enumerate(dataloader):

        size_of_data+=image_bplusc.shape[0]

        if (torch.cuda.is_available()):

            image_bplusc = image_bplusc.cuda()
            image_atimesb = image_atimesb.cuda()
            image_atimesc = image_atimesc.cuda()
            image_btimesa = image_btimesa.cuda()
            image_ctimesa = image_ctimesa.cuda()
            image_1 = image_1.cuda()
            image_2 = image_2.cuda()
            image_3 = image_3.cuda()

        # getting predictions and probabilities from sum and single model
        logits_sum_23 = sum_model(image_bplusc)
        probs_sum_23 = torch.nn.functional.softmax(logits_sum_23, dim=1)
        logits_prod_12 = prod_model(image_atimesb)
        probs_prod_12 = torch.nn.functional.softmax(logits_prod_12, dim=1)
        logits_prod_13 = prod_model(image_atimesc)
        probs_prod_13 = torch.nn.functional.softmax(logits_prod_13 , dim=1)
        logits_prod_21 = prod_model(image_btimesa)
        probs_prod_21 = torch.nn.functional.softmax(logits_prod_21, dim=1)
        logits_prod_31 = prod_model(image_ctimesa)
        probs_prod_31 = torch.nn.functional.softmax(logits_prod_31, dim=1)

        # getting predicted labels from batch
        _, pred_labels_sum_23 = torch.max(probs_sum_23, dim=1)
        _, pred_labels_prod_12 = torch.max(probs_prod_12, dim=1)
        _, pred_labels_prod_13 = torch.max(probs_prod_13, dim=1)
        _, pred_labels_prod_21 = torch.max(probs_prod_21, dim=1)
        _, pred_labels_prod_31 = torch.max(probs_prod_31, dim=1)

        # create batch of images (from test_images_MNIST) corresponding to pred labels
        im_pred_labels_sum_23 = images_from_pred_labels(pred_labels_sum_23, dev_images_MNIST)
        im_pred_labels_prod_12 = images_from_pred_labels(pred_labels_prod_12, dev_images_MNIST)
        im_pred_labels_prod_13 = images_from_pred_labels(pred_labels_prod_13, dev_images_MNIST)
        im_pred_labels_prod_21 = images_from_pred_labels(pred_labels_prod_21, dev_images_MNIST)
        im_pred_labels_prod_31 = images_from_pred_labels(pred_labels_prod_31, dev_images_MNIST)

        # concatenating images for distributivity (in batches)
        leftdist_1_times_2plus3 = torch.cat((image_1,im_pred_labels_sum_23), dim=3)
        leftdist_1prod2_plus_1prod3 = torch.cat((im_pred_labels_prod_12, im_pred_labels_prod_13), dim=3)
        rightdist_2plus3_times_1 = torch.cat((im_pred_labels_sum_23,image_1), dim=3)
        rightdist_2prod1_plus_3prod1 = torch.cat((im_pred_labels_prod_21, im_pred_labels_prod_31), dim=3)

        # predictions logits and probabilities for associative combinations
        pred_leftdist_1_times_2plus3 = prod_model(leftdist_1_times_2plus3)
        probs_leftdist_1_times_2plus3 = torch.nn.functional.softmax(pred_leftdist_1_times_2plus3, dim=1)
        pred_leftdist_1prod2_plus_1prod3 = sum_model(leftdist_1prod2_plus_1prod3)
        probs_leftdist_1prod2_plus_1prod3 = torch.nn.functional.softmax(pred_leftdist_1prod2_plus_1prod3, dim=1)
        pred_rightdist_2plus3_times_1 = prod_model(rightdist_2plus3_times_1)
        probs_rightdist_2plus3_times_1 = torch.nn.functional.softmax(pred_rightdist_2plus3_times_1, dim=1)
        pred_rightdist_2prod1_plus_3prod1 = sum_model(rightdist_2prod1_plus_3prod1)
        probs_rightdist_2prod1_plus_3prod1 = torch.nn.functional.softmax(pred_rightdist_2prod1_plus_3prod1, dim=1)

        # getting predicted labels from batch
        _, pred_labels_leftdist_1_times_2plus3 = torch.max(probs_leftdist_1_times_2plus3, dim=1)
        _, pred_labels_leftdist_1prod2_plus_1prod3 = torch.max(probs_leftdist_1prod2_plus_1prod3, dim=1)
        _, pred_labels_rightdist_2plus3_times_1 = torch.max(probs_rightdist_2plus3_times_1, dim=1)
        _, pred_labels_rightdist_2prod1_plus_3prod1 = torch.max(probs_rightdist_2prod1_plus_3prod1, dim=1)

        ld_lhs = torch.unsqueeze(pred_labels_leftdist_1_times_2plus3,0)
        # Left Dist LHS
        for i in range(k_sample):

            im_pred_labels_sum_23 = images_from_pred_labels(pred_labels_sum_23, dev_images_MNIST)
            leftdist_1_times_2plus3 = torch.cat((image_1,im_pred_labels_sum_23), dim=3)
            pred_leftdist_1_times_2plus3 = prod_model(leftdist_1_times_2plus3)
            probs_leftdist_1_times_2plus3 = torch.nn.functional.softmax(pred_leftdist_1_times_2plus3, dim=1)
            _, pred_labels_leftdist_1_times_2plus3 = torch.max(probs_leftdist_1_times_2plus3, dim=1)

            ld_lhs =  torch.cat((ld_lhs, torch.unsqueeze(pred_labels_leftdist_1_times_2plus3,0)), dim=0)

        ld_lhs_final = torch.transpose(ld_lhs,0,1)
        pred_lab_ld_lhs = majority_labels(ld_lhs_final)

        ld_rhs = torch.unsqueeze(pred_labels_leftdist_1prod2_plus_1prod3, 0)
        # Left Dist RHS
        for i in range(k_sample):
            im_pred_labels_prod_12 = images_from_pred_labels(pred_labels_prod_12, dev_images_MNIST)
            for j in range(k_sample):
                im_pred_labels_prod_13 = images_from_pred_labels(pred_labels_prod_13, dev_images_MNIST)
                leftdist_1prod2_plus_1prod3 = torch.cat((im_pred_labels_prod_12, im_pred_labels_prod_13), dim=3)
                pred_leftdist_1prod2_plus_1prod3 = sum_model(leftdist_1prod2_plus_1prod3)
                probs_leftdist_1prod2_plus_1prod3 = torch.nn.functional.softmax(pred_leftdist_1prod2_plus_1prod3, dim=1)
                _, pred_labels_leftdist_1prod2_plus_1prod3 = torch.max(probs_leftdist_1prod2_plus_1prod3, dim=1)

                ld_rhs = torch.cat((ld_rhs, torch.unsqueeze(pred_labels_leftdist_1prod2_plus_1prod3 ,0)), dim=0)

        ld_rhs_final = torch.transpose(ld_rhs,0,1)
        pred_lab_ld_rhs = majority_labels(ld_rhs_final)

        # Right Dist LHS
        rd_lhs = torch.unsqueeze(pred_labels_rightdist_2plus3_times_1, 0)
        for i in range(k_sample):

            im_pred_labels_sum_23 = images_from_pred_labels(pred_labels_sum_23, dev_images_MNIST)
            rightdist_2plus3_times_1 = torch.cat((im_pred_labels_sum_23,image_1), dim=3)
            pred_rightdist_2plus3_times_1 = prod_model(rightdist_2plus3_times_1)
            probs_rightdist_2plus3_times_1 = torch.nn.functional.softmax(pred_rightdist_2plus3_times_1, dim=1)
            _, pred_labels_rightdist_2plus3_times_1 = torch.max(probs_rightdist_2plus3_times_1, dim=1)

            rd_lhs = torch.cat((rd_lhs,torch.unsqueeze(pred_labels_rightdist_2plus3_times_1, 0)), dim=0)

        rd_lhs_final = torch.transpose(rd_lhs,0,1)
        pred_lab_rd_lhs = majority_labels(rd_lhs_final)

        # Right Dist RHS
        rd_rhs = torch.unsqueeze(pred_labels_rightdist_2prod1_plus_3prod1, 0)
        for i in range(k_sample):
            im_pred_labels_prod_21 = images_from_pred_labels(pred_labels_prod_21, dev_images_MNIST)
            for j in range(k_sample):
                im_pred_labels_prod_31 = images_from_pred_labels(pred_labels_prod_31, dev_images_MNIST)
                rightdist_2prod1_plus_3prod1 = torch.cat((im_pred_labels_prod_21, im_pred_labels_prod_31), dim=3)
                pred_rightdist_2prod1_plus_3prod1 = sum_model(rightdist_2prod1_plus_3prod1)
                probs_rightdist_2prod1_plus_3prod1 = torch.nn.functional.softmax(pred_rightdist_2prod1_plus_3prod1, dim=1)
                _, pred_labels_rightdist_2prod1_plus_3prod1 = torch.max(probs_rightdist_2prod1_plus_3prod1, dim=1)

                rd_rhs = torch.cat((rd_rhs, torch.unsqueeze(pred_labels_rightdist_2prod1_plus_3prod1, 0)), dim=0)

        rd_rhs_final = torch.transpose(rd_rhs,0,1)
        pred_lab_rd_rhs = majority_labels(rd_rhs_final)

        # calculating averaged difference between f_0 and f_1
        equals_left_distributivity = (pred_lab_ld_lhs == pred_lab_ld_rhs)
        total_left_distributivity += int(torch.sum(equals_left_distributivity.type(torch.FloatTensor)).data.cpu())

        equals_right_distributivity = (pred_lab_rd_lhs == pred_lab_rd_rhs)
        total_right_distributivity += int(torch.sum(equals_right_distributivity.type(torch.FloatTensor)).data.cpu())


    return total_left_distributivity / size_of_data, total_right_distributivity / size_of_data






