import torch
import random
import numpy as np
from torch import nn
from model import OpeModuloModel, Model
import argparse



def training_mnist(model=None, optim=1, lr=0.001, n_epochs=100, train_dataloader=None, val_dataloader=None, lenmnistval=0, name=None, tnorm=None, test=True):

    model=model
    
    use_sigmoid=False

    if use_sigmoid:
        criterion = torch.nn.MultiLabelSoftMarginLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if optim==1:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optim==0:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)

    if (torch.cuda.is_available()):
        model.cuda()

    no_epochs = n_epochs
    train_loss = list()
    val_loss = list()
    num_updates = 0.0

    best_accuracy = 0.0
    best_ep = 0
    for epoch in range(no_epochs):
       
        total_train_loss = 0
        total_val_loss = 0
        h = 0
        model.train()
        
        # training
        for itr, (image, label) in enumerate(train_dataloader):

            current_batch_size = image.shape[0]
            h += current_batch_size

            if (torch.cuda.is_available()):
                image = image.cuda()
                label = label.cuda()

            optimizer.zero_grad()

            pred = model(image)

            if tnorm == 'luka':

                if use_sigmoid:
                    image_probs = sigmoid(pred)
                else:
                    image_probs = softmax(pred)
                
                loss = -torch.sum(torch.gather(image_probs, 1, torch.unsqueeze(label, -1)))


            if tnorm == 'prod' or tnorm == 'rprod':

                if use_sigmoid:
                    labels_one_hot = torch.nn.functional.one_hot(label, num_classes=10) #Shape (B, C)
                    loss = criterion(pred, labels_one_hot)
                else:
                    loss = criterion(pred, label)
            
            if tnorm == 'godel' or tnorm == 'rgodel':
                
                image_probs = softmax(pred)
                
                correct_class_probs = torch.gather(image_probs, 1, torch.unsqueeze(label, -1))
                
                if 4 == 3:
                    #loss = -torch.sum(torch.gather(image_probs, 1, torch.unsqueeze(label, -1)))
                    loss = criterion(pred, label)
                else:
                
                    loss = -1 * torch.min(correct_class_probs)
            
            
            total_train_loss += loss.item()

            loss.backward()
            num_updates +=1
            optimizer.step()
            

        total_train_loss = total_train_loss / (itr + 1)
        train_loss.append(total_train_loss)

        # validation
        model.eval()
        total = 0
        for itr, (image, label) in enumerate(val_dataloader):

            if (torch.cuda.is_available()):
                image = image.cuda()
                label = label.cuda()

            pred = model(image)
        
            if use_sigmoid:
                labels_one_hot = torch.nn.functional.one_hot(label, num_classes=10) #Shape (B, C)
                loss = criterion(pred, labels_one_hot)
            else:
                #criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(pred, label)

            total_val_loss += loss.item()

            if use_sigmoid:
                pred = torch.nn.functional.sigmoid(pred)
            else:
                pred = torch.nn.functional.softmax(pred, dim=1)
            
            for i, p in enumerate(pred):
                if label[i] == torch.max(p.data, 0)[1]:
                    total = total + 1

        accuracy = total / lenmnistval

        total_val_loss = total_val_loss / h
        val_loss.append(total_val_loss)

        print('\nEpoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}, Num Updates: {}'.format(epoch + 1, no_epochs, total_train_loss, total_val_loss, accuracy, num_updates))

        if best_accuracy < accuracy:
            best_ep = epoch+1
            best_accuracy = accuracy
            saved_as = name
            if test!=True:
                torch.save(model.state_dict(), saved_as)
    
    print('best accuracy', best_accuracy, 'best epoch', best_ep)
    return best_accuracy, saved_as


