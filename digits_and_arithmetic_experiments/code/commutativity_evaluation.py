import torch
import pickle
import sys
import random
from model import OpeModuloModel, Model
import torchvision.datasets as datasets
from torchvision import transforms



def commutativity(operator_model, dataloader):

    operator_model.eval()
    total_commutativity = 0
    size_of_data = 0

    for itr, (img_1, img_2, cat_img, cat_reverse_img)  in enumerate(dataloader):

        size_of_data += img_1.shape[0]

        if (torch.cuda.is_available()):
            img_1 = img_1.cuda()
            img_2 = img_2.cuda()
            cat_img = cat_img.cuda()
            cat_reverse_img = cat_reverse_img.cuda()

        # logits and probs for SUM operation
        logits_cat_op_img = operator_model(cat_img)
        cat_op_probs = torch.nn.functional.softmax(logits_cat_op_img, dim=1)
        
        # logist and probs for PROD operation
        logits_cat_rev_op_img = operator_model(cat_reverse_img)
        cat_rev_op_probs = torch.nn.functional.softmax(logits_cat_rev_op_img, dim=1)

        # getting predicted labels from batch
        _, pred_labels_cat = torch.max(cat_op_probs, dim=1)
        _, pred_labels_rev_cat = torch.max(cat_rev_op_probs, dim=1)
        
        # COMMUTATIVITY
        equals_commut = (pred_labels_cat == pred_labels_rev_cat)
        total_commutativity += int(torch.sum(equals_commut.type(torch.FloatTensor)).data.cpu())

    return total_commutativity / size_of_data
            
