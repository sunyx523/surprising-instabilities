'''Computes result for Table 1 in section 4.2'''
import numpy as np

r = 1 #random seed
loss1 = np.load('../result/parameter/accuracy/rtest' + str(r) + '_cifar10_resnet_sgd_1.npy')
loss2 = np.load('../result/parameter/accuracy/test' + str(r) + '_cifar10_resnet_sgd_2.npy')
loss3 = np.load('../result/parameter/accuracy/test' + str(r) + '_cifar10_resnet_sgd_3.npy')
loss4 = np.load('../result/parameter/accuracy/test' + str(r) + '_cifar10_resnet_sgd_4.npy')
loss5 = np.load('../result/parameter/accuracy/test' + str(r) + '_cifar10_resnet_sgd_5.npy')
loss6 = np.load('../result/parameter/accuracy/test' + str(r) + '_cifar10_resnet_sgd_6.npy')

#Compute the average accuracy for last ten epochs
l1 = np.mean(loss1[-10:])
l2 = np.mean(loss2[-10:])
l3 = np.mean(loss3[-10:])
l4 = np.mean(loss4[-10:])
l5 = np.mean(loss5[-10:])
l6 = np.mean(loss6[-10:])
l = [l1, l2, l3, l4, l5, l6]

#Print result
print(l1)
print(l2)
print(l3)
print(l4)
print(l5)
print(l6)
print(np.std(l))
