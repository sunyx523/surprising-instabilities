'''Computes result for Table 1 in section 4.2'''
import numpy as np

result_root = '../result/accuracy/cifar10_resnet_sgd/'
mode = 'seed' #'seed' | 'perturb'

if mode == 'seed':
    r = 1 #random seed
    loss1 = np.load(result_root + '/seed' + str(r) + '/test1.npy')
    loss2 = np.load(result_root + '/seed' + str(r) + '/test2.npy')
    loss3 = np.load(result_root + '/seed' + str(r) + '/test3.npy')
    loss4 = np.load(result_root + '/seed' + str(r) + '/test4.npy')
    loss5 = np.load(result_root + '/seed' + str(r) + '/test5.npy')
    loss6 = np.load(result_root + '/seed' + str(r) + '/test6.npy')
elif mode == 'perturb':
    n = 1 #perturbation
    loss1 = np.load(result_root + '/seed1/test' + str(n) + '.npy')
    loss2 = np.load(result_root + '/seed2/test' + str(n) + '.npy')
    loss3 = np.load(result_root + '/seed3/test' + str(n) + '.npy')
    loss4 = np.load(result_root + '/seed4/test' + str(n) + '.npy')
    loss5 = np.load(result_root + '/seed5/test' + str(n) + '.npy')
    loss6 = np.load(result_root + '/seed6/test' + str(n) + '.npy')
else:
    raise Exception("Wrong mode")

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
