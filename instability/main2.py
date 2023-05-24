'''Code for Figure 2 right in Section 4'''
import os
import random
import time
import copy
import argparse
import sys
import numpy as np
import numpy.matlib
import scipy
from scipy.io import savemat

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.nn import init
from torch.autograd import Variable
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--r', type=int, default=7, help='random seed')
parser.add_argument('--data_root', type=str, default='../data_Cifar10', help='dataset root')
parser.add_argument('--batchsize_test', type=int, default=256, help="testing batch size")
parser.add_argument('--batchsize_train', type=int, default=128, help='training batch szie')
parser.add_argument('--lr0', type=float, default=0.1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0, help='momentum')
parser.add_argument('--Nesterov', type=bool, default=False, help='Nesterov')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--n', type=int, default=1, help='inital learning rate')
parser.add_argument('--result_root', type=str, default='../parameter/iteration/', help='result root')
args = parser.parse_args()

# Initialize random seed
torch.manual_seed(args.r) # cpu
torch.cuda.manual_seed(args.r) #gpu
torch.backends.cudnn.benchmark=False# cudnn
torch.backends.cudnn.deterministic=True
#torch.set_default_dtype(torch.float64)

from resnet_swish import *
from PDE_nesterov import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    """
    The main function.
    """
    use_cuda = torch.cuda.is_available()
    global best_acc
    best_acc = 0
    start_epoch = 0

    # Load the Cifar10 data
    print('==> Preparing data...')
    
    train_set = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]))
    
    test_set = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]))
       
    print('Batch size of the test set: ', args.batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=args.batchsize_test,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True
                                             )
    
    print('Batch size of the train set: ', args.batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=args.batchsize_train,
                                               shuffle=True,
                                               num_workers=1,
                                               pin_memory=True
                                              )
    
    print('Total training (known) batch number: ', len(train_loader))
    print('Total testing batch number: ', len(test_loader))
    
    #--------------------------------------------------------------------------
    # Build the model
    #--------------------------------------------------------------------------
    net1 = resnet56(num_classes=10).to(device)
    net2 = resnet56(num_classes=10).to(device)
    net2.load_state_dict(net1.state_dict())
    criterion = nn.CrossEntropyLoss()

    address1 = args.result_root + 'swish_' + str(args.lr0) + '/L' + str(args.weight_decay) + '_' + str(args.n) + '_1/'
    address2 = args.result_root + 'swish_' + str(args.lr0) + '/L' + str(args.weight_decay) + '_' + str(args.n) + '_2/'
    if not os.path.exists(address1):
        os.makedirs(address1)
    if not os.path.exists(address2):
        os.makedirs(address2)
    
    lr = args.lr0 / args.n
    optimizer1 = PDE_nesterov(net1.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.Nesterov, k = 1)
    optimizer2 = PDE_nesterov(net2.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.Nesterov, k = 3)
   
    test_loss_list1 = []
    test_accuracy_list1 = []
    test_loss_list2 = []
    test_accuracy_list2 = []

    param_id = 1
    it_id = 1
    nepoch = args.n
    for epoch in range(nepoch):
        print('Epoch ID: ', epoch)
        
        # Training
        correct1 = 0.; total1 = 0.; train_loss1 = 0.; 
        correct2 = 0.; total2 = 0.; train_loss2 = 0.;
        net1.train()
        net2.train()

        for batch_idx, (x, target) in enumerate(train_loader):
            it_id = it_id + 1
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            x, target = x.to(device), target.to(device)
            score1 = net1(x)
            score2 = net2(x)
            loss1 = criterion(score1, target)
            loss2 = criterion(score2, target)
            loss1.backward()
            loss2.backward()
            optimizer1.step()
            optimizer2.step()
            
            train_loss1 += loss1.data.item()
            train_loss2 += loss2.data.item()
            total1 += target.size(0)
            total2 += target.size(0)
            _, predicted1 = torch.max(score1.data, 1)
            _, predicted2 = torch.max(score2.data, 1)
            correct1 += predicted1.eq(target.data).cpu().sum()
            correct2 += predicted2.eq(target.data).cpu().sum()

            # Save the weights
            for pname, p in net1.named_parameters():
                if 'layer2.5.conv1.weight' in pname and it_id % (args.n) == 0:
                    scipy.io.savemat(address1 + str(param_id) + '_conv_weight.mat', {'conv_weight': p.data.cpu().numpy()})
        
            for pname, p in net2.named_parameters():
                if 'layer2.5.conv1.weight' in pname and it_id % (args.n) == 0:
                    scipy.io.savemat(address2 + str(param_id) + '_conv_weight.mat', {'conv_weight': p.data.cpu().numpy()})
                    param_id = param_id + 1

        # Testing
        test_loss1 = 0.; correct1 = 0.; total1 = 0.
        test_loss2 = 0.; correct2 = 0.; total2 = 0.
        net1.eval()
        net2.eval()
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_loader):
                x, target = x.to(device), target.to(device)
                score1 = net1(x)
                score2 = net2(x)
                loss1 = criterion(score1, target)
                loss2 = criterion(score2, target)
                
                test_loss1 += loss1.data.item()
                test_loss2 += loss2.data.item()
                _, predicted1 = torch.max(score1.data, 1)
                _, predicted2 = torch.max(score2.data, 1)
                total1 += target.size(0)
                total2 += target.size(0)
                correct1 += predicted1.eq(target.data).cpu().sum()
                correct2 += predicted2.eq(target.data).cpu().sum()

            test_loss_list1.append(test_loss1/(batch_idx+1))
            test_loss_list2.append(test_loss2/(batch_idx+1))
            test_accuracy_list1.append(100.*correct1/total1)
            test_accuracy_list2.append(100.*correct2/total2)

        acc1 = 100.*correct1/total1
        acc2 = 100.*correct2/total2
        print(acc1.numpy(), '', acc2.numpy())

