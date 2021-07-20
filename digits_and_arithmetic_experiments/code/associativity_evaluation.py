import torch
import pickle
import sys
import random
from model import OpeModuloModel, Model
import torchvision.datasets as datasets
from torchvision import transforms
import argparse


# we select a random image of the predicted digit form the dev set
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

# we output the label that was predicted more times among all the samples
def majority_labels(cato):
    
    final = torch.empty([cato.shape[0]])
    i=0
    for row in cato:

        rc = torch.stack([(row==x_u).sum() for x_u in list(range(10))])
        _,index = torch.max(rc, dim=0)
        #print('ind', index)
        final[i]=index
        i+=1
    return final




def associativity(operator_model, dataloader, test_images_MNIST):
    
    operator_model.eval()

    k_samples = 5
    total = 0
    size_of_data = 0

    for itr, (image_1, image_2, image_3, new_concat_im1_im2, new_concat_im2_im3)  in enumerate(dataloader):

        size_of_data+=image_1.shape[0]
        
        if (torch.cuda.is_available()):
            image_1 = image_1.cuda()
            image_2 = image_2.cuda()
            image_3 = image_3.cuda()
            new_concat_im1_im2 = new_concat_im1_im2.cuda()
            new_concat_im2_im3 = new_concat_im2_im3.cuda()
            
        # getting logits and probabilities from operator model
        logits_ope_12 = operator_model(new_concat_im1_im2)
        probs_ope_12 = torch.nn.functional.softmax(logits_ope_12, dim=1)
        logits_ope_23 = operator_model(new_concat_im2_im3)
        probs_ope_23 = torch.nn.functional.softmax(logits_ope_23, dim=1)
        
        # getting predicted labels from batch
        _, pred_labels_ope_12 = torch.max(probs_ope_12,dim=1)
        _, pred_labels_ope_23 = torch.max(probs_ope_23,dim=1)
       

        # create batch of images (from test_images_MNIST) corresponding to pred labels
        im_pred_labels_ope_12 = images_from_pred_labels(pred_labels_ope_12, test_images_MNIST)
        im_pred_labels_ope_23 = images_from_pred_labels(pred_labels_ope_23,test_images_MNIST)
    
        # concatenating images for associativity (in batches)
        assoc_12_3 = torch.cat((im_pred_labels_ope_12,image_3), dim=3) 
        assoc_1_23 = torch.cat((image_1, im_pred_labels_ope_23), dim=3)

        # logits and probabilities for associative combinations
        logits_assoc_12_3 = operator_model(assoc_12_3)
        probs_assoc_12_3 = torch.nn.functional.softmax(logits_assoc_12_3, dim=1) 
        logits_assoc_1_23 = operator_model(assoc_1_23) 
        probs_assoc_1_23 = torch.nn.functional.softmax(logits_assoc_1_23, dim=1) # NEW LINE

        # getting predicted labels from batch
        _, pred_labels_ope_12_3 = torch.max(probs_assoc_12_3,dim=1) 
        _, pred_labels_ope_1_23 = torch.max(probs_assoc_1_23, dim=1) 
 
        
        cato1 = torch.unsqueeze(pred_labels_ope_12_3,0)
        cato2 = torch.unsqueeze(pred_labels_ope_1_23,0)

        for i in range(k_samples):

            # create batch of images (from test_images_MNIST) corresponding to pred labels
            im_pred_labels_ope_12 = images_from_pred_labels(pred_labels_ope_12, test_images_MNIST)
            im_pred_labels_ope_23 = images_from_pred_labels(pred_labels_ope_23,test_images_MNIST)
            # concatenating images for associativity (in batches)
            assoc_12_3 = torch.cat((im_pred_labels_ope_12,image_3), dim=3)
            assoc_1_23 = torch.cat((image_1, im_pred_labels_ope_23), dim=3)
            # logits and probabilities for associative combinations
            logits_assoc_12_3 = operator_model(assoc_12_3)
            probs_assoc_12_3 = torch.nn.functional.softmax(logits_assoc_12_3, dim=1) 
            logits_assoc_1_23 = operator_model(assoc_1_23)
            probs_assoc_1_23 = torch.nn.functional.softmax(logits_assoc_1_23, dim=1)
            # getting predicted labels from batch
            _, pred_labels_ope_12_3 = torch.max(probs_assoc_12_3,dim=1)
            _, pred_labels_ope_1_23 = torch.max(probs_assoc_1_23, dim=1)
            
            cato1 = torch.cat((cato1, torch.unsqueeze(pred_labels_ope_12_3,0)), dim=0)
            cato2 = torch.cat((cato2, torch.unsqueeze(pred_labels_ope_1_23,0)), dim=0)

        cato1_final = torch.transpose(cato1,0,1)
        cato2_final = torch.transpose(cato2,0,1)

        pred_labels_ope_12_3 = majority_labels(cato1_final)
        pred_labels_ope_1_23 = majority_labels(cato2_final)

        equals_12_3_1_23 = (pred_labels_ope_12_3 == pred_labels_ope_1_23) 
    
        total += int(torch.sum(equals_12_3_1_23.type(torch.FloatTensor)).data.cpu()) 


    return total / size_of_data





