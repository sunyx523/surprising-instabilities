'''Code for Table 1 left in Section 4'''
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--r', type=int, default=7, help='random seed')
parser.add_argument('--data_root', type=str, default='../data_Cifar10', help='dataset root')
parser.add_argument('--batchsize_test', type=int, default=256, help="testing batch size")
parser.add_argument('--batchsize_train', type=int, default=128, help='training batch szie')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--Nesterov', type=bool, default=True, help='Nesterov')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--nepoch', type=int, default=200, help='total epoch number')
parser.add_argument('--result_root', type=str, default='../result/accuracy/cifar10_resnet_sgd/', help='result root')
args = parser.parse_args()

from resnet import * #ReLU activation
# from resnet_swish import * #Swish activation
from PDE_nesterov import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.r)  # cpu
torch.cuda.manual_seed(args.r)  # gpu
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  # cudnn
# torch.set_default_dtype(torch.bfloat16)

if __name__ == '__main__':
    
    use_cuda = torch.cuda.is_available()
    global best_acc
    best_acc = 0
    start_epoch = 0
    criterion = nn.CrossEntropyLoss()
    

    # Load the data
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
    
    # Build the model
    net1 = resnet56(num_classes=10).to(device)
    net2 = copy.deepcopy(net1)
    net3 = copy.deepcopy(net1)
    net4 = copy.deepcopy(net1)
    net5 = copy.deepcopy(net1)
    net6 = copy.deepcopy(net1)

    optimizer1 = PDE_nesterov(net1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.Nesterov, k = 1)
    optimizer2 = PDE_nesterov(net2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.Nesterov, k = 3)
    optimizer3 = PDE_nesterov(net3.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.Nesterov, k = 5)
    optimizer4 = PDE_nesterov(net4.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.Nesterov, k = 7)
    optimizer5 = PDE_nesterov(net5.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.Nesterov, k = 9)
    optimizer6 = PDE_nesterov(net6.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.Nesterov, k = 11)

    train_loss_list1 = []
    train_accuarcy_list1 = []
    test_loss_list1 = []
    test_accuarcy_list1 = []

    train_loss_list2 = []
    train_accuarcy_list2 = []
    test_loss_list2 = []
    test_accuarcy_list2 = []

    train_loss_list3 = []
    train_accuarcy_list3 = []
    test_loss_list3 = []
    test_accuarcy_list3 = []

    train_loss_list4 = []
    train_accuarcy_list4 = []
    test_loss_list4 = []
    test_accuarcy_list4 = []

    train_loss_list5 = []
    train_accuarcy_list5 = []
    test_loss_list5 = []
    test_accuarcy_list5 = []

    train_loss_list6 = []
    train_accuarcy_list6 = []
    test_loss_list6 = []
    test_accuarcy_list6 = []

    lr = args.lr
    for epoch in range(args.nepoch):
        print('Epoch ID: ', epoch)
        
        # Training
        if epoch >= 40 and (epoch//40 == epoch/40.0):
            lr = lr/10
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr

            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr

            for param_group in optimizer3.param_groups:
                param_group['lr'] = lr

            for param_group in optimizer4.param_groups:
                param_group['lr'] = lr

            for param_group in optimizer5.param_groups:
                param_group['lr'] = lr

            for param_group in optimizer6.param_groups:
                param_group['lr'] = lr

        correct1 = 0.; train_loss1 = 0.; total = 0.;
        correct2 = 0.; train_loss2 = 0.;
        correct3 = 0.; train_loss3 = 0.;
        correct4 = 0.; train_loss4 = 0.; 
        correct5 = 0.; train_loss5 = 0.; 
        correct6 = 0.; train_loss6 = 0.; 
        net1.train()
        net2.train()
        net3.train()
        net4.train()
        net5.train()
        net6.train()
        loss_accuarcy_epoch0_list1 = []
        loss_accuarcy_epoch0_list2 = []
        loss_accuarcy_epoch0_list3 = []
        loss_accuarcy_epoch0_list4 = []
        loss_accuarcy_epoch0_list5 = []
        loss_accuarcy_epoch0_list6 = []

        for batch_idx, (x, target) in enumerate(train_loader):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            optimizer5.zero_grad()
            optimizer6.zero_grad()
            x, target = x.to(device), target.to(device)
            score1 = net1(x)
            score2 = net2(x)
            score3 = net3(x)
            score4 = net4(x)
            score5 = net5(x)
            score6 = net6(x)
            loss1 = criterion(score1, target)
            loss2 = criterion(score2, target)
            loss3 = criterion(score3, target)
            loss4 = criterion(score4, target)
            loss5 = criterion(score5, target)
            loss6 = criterion(score6, target)
            loss1.backward()
            loss2.backward()
            loss3.backward()
            loss4.backward()
            loss5.backward()
            loss6.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()
            optimizer6.step()
            
            train_loss1 += loss1.data.item()
            train_loss2 += loss2.data.item()
            train_loss3 += loss3.data.item()
            train_loss4 += loss4.data.item()
            train_loss5 += loss5.data.item()
            train_loss6 += loss6.data.item()
            _, predicted1 = torch.max(score1.data, 1)
            _, predicted2 = torch.max(score2.data, 1)
            _, predicted3 = torch.max(score3.data, 1)
            _, predicted4 = torch.max(score4.data, 1)
            _, predicted5 = torch.max(score5.data, 1)
            _, predicted6 = torch.max(score6.data, 1)
            total += target.size(0)

            loss_accuarcy_epoch0_list1.append(loss1.data.item())
            loss_accuarcy_epoch0_list2.append(loss2.data.item())
            loss_accuarcy_epoch0_list3.append(loss3.data.item())
            loss_accuarcy_epoch0_list4.append(loss4.data.item())
            loss_accuarcy_epoch0_list5.append(loss5.data.item())
            loss_accuarcy_epoch0_list6.append(loss6.data.item())

            correct1 += predicted1.eq(target.data).cpu().sum()
            correct2 += predicted2.eq(target.data).cpu().sum()
            correct3 += predicted3.eq(target.data).cpu().sum()
            correct4 += predicted4.eq(target.data).cpu().sum()
            correct5 += predicted5.eq(target.data).cpu().sum()
            correct6 += predicted6.eq(target.data).cpu().sum()
     
        train_loss_list1.append(train_loss1/(batch_idx+1))
        train_loss_list2.append(train_loss2/(batch_idx+1))
        train_loss_list3.append(train_loss3/(batch_idx+1))
        train_loss_list4.append(train_loss4/(batch_idx+1))
        train_loss_list5.append(train_loss5/(batch_idx+1))
        train_loss_list6.append(train_loss6/(batch_idx+1))
        train_accuarcy_list1.append(100.*correct1/total)
        train_accuarcy_list2.append(100.*correct2/total)
        train_accuarcy_list3.append(100.*correct3/total)
        train_accuarcy_list4.append(100.*correct4/total)
        train_accuarcy_list5.append(100.*correct5/total)
        train_accuarcy_list6.append(100.*correct6/total)
              
        # Testing
        test_loss1 = 0.; correct1 = 0.; total = 0.
        test_loss2 = 0.; correct2 = 0.; 
        test_loss3 = 0.; correct3 = 0.; 
        test_loss4 = 0.; correct4 = 0.; 
        test_loss5 = 0.; correct5 = 0.; 
        test_loss6 = 0.; correct6 = 0.; 
        net1.eval()
        net2.eval()
        net3.eval()
        net4.eval()
        net5.eval()
        net6.eval()
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_loader):
                x, target = x.to(device), target.to(device)
                score1 = net1(x)
                score2 = net2(x)
                score3 = net3(x)
                score4 = net4(x)
                score5 = net5(x)
                score6 = net6(x)
                loss1 = criterion(score1, target)
                loss2 = criterion(score2, target)
                loss3 = criterion(score3, target)
                loss4 = criterion(score4, target)
                loss5 = criterion(score5, target)
                loss6 = criterion(score6, target)
                
                test_loss1 += loss1.data.item()
                test_loss2 += loss2.data.item()
                test_loss3 += loss3.data.item()
                test_loss4 += loss4.data.item()
                test_loss5 += loss5.data.item()
                test_loss6 += loss6.data.item()
                _, predicted1 = torch.max(score1.data, 1)
                _, predicted2 = torch.max(score2.data, 1)
                _, predicted3 = torch.max(score3.data, 1)
                _, predicted4 = torch.max(score4.data, 1)
                _, predicted5 = torch.max(score5.data, 1)
                _, predicted6 = torch.max(score6.data, 1)
                total += target.size(0)
                correct1 += predicted1.eq(target.data).cpu().sum()
                correct2 += predicted2.eq(target.data).cpu().sum()
                correct3 += predicted3.eq(target.data).cpu().sum()
                correct4 += predicted4.eq(target.data).cpu().sum()
                correct5 += predicted5.eq(target.data).cpu().sum()
                correct6 += predicted6.eq(target.data).cpu().sum()

            test_loss_list1.append(test_loss1/(batch_idx+1))
            test_loss_list2.append(test_loss2/(batch_idx+1))
            test_loss_list3.append(test_loss3/(batch_idx+1))
            test_loss_list4.append(test_loss4/(batch_idx+1))
            test_loss_list5.append(test_loss5/(batch_idx+1))
            test_loss_list6.append(test_loss6/(batch_idx+1))
            test_accuarcy_list1.append(100.*correct1/total)
            test_accuarcy_list2.append(100.*correct2/total)
            test_accuarcy_list3.append(100.*correct3/total)
            test_accuarcy_list4.append(100.*correct4/total)
            test_accuarcy_list5.append(100.*correct5/total)
            test_accuarcy_list6.append(100.*correct6/total)        

        acc1 = 100.*correct1/total
        acc2 = 100.*correct2/total
        acc3 = 100.*correct3/total
        acc4 = 100.*correct4/total
        acc5 = 100.*correct5/total
        acc6 = 100.*correct6/total
        print(acc1.numpy(), '', acc2.numpy(), ' ', acc3.numpy(), ' ', acc4.numpy(), ' ', acc5.numpy(), ' ', acc6.numpy())
        
        # Save the test accuracy
        np.save(args.result_root + '/seed' + str(args.r) + '/test1.npy', test_accuarcy_list1)
        np.save(args.result_root + '/seed' + str(args.r) + '/test2.npy', test_accuarcy_list2)
        np.save(args.result_root + '/seed' + str(args.r) + '/test3.npy', test_accuarcy_list3)
        np.save(args.result_root + '/seed' + str(args.r) + '/test4.npy', test_accuarcy_list4)
        np.save(args.result_root + '/seed' + str(args.r) + '/test5.npy', test_accuarcy_list5)
        np.save(args.result_root + '/seed' + str(args.r) + '/test6.npy', test_accuarcy_list6)
        np.save(args.result_root + '/seed' + str(args.r) + '/train1.npy', train_accuarcy_list1)
        np.save(args.result_root + '/seed' + str(args.r) + '/train2.npy', train_accuarcy_list2)
        np.save(args.result_root + '/seed' + str(args.r) + '/train3.npy', train_accuarcy_list3)
        np.save(args.result_root + '/seed' + str(args.r) + '/train4.npy', train_accuarcy_list4)
        np.save(args.result_root + '/seed' + str(args.r) + '/train5.npy', train_accuarcy_list5)
        np.save(args.result_root + '/seed' + str(args.r) + '/train6.npy', train_accuarcy_list6)
