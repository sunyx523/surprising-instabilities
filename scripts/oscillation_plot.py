'''Plots Figure 4 left in section 5.4'''
import numpy as np
import os.path
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


loss1 = np.load('../result/loss/nonlinear/loss_dt0.01_L20.npy')
loss3 = np.load('../result/loss/nonlinear/loss_dt0.03_L20.npy')
loss5 = np.load('../result/loss/nonlinear/loss_dt0.05_L20.npy')
loss7 = np.load('../result/loss/nonlinear/loss_dt0.07_L20.npy')
loss9 = np.load('../result/loss/nonlinear/loss_dt0.09_L20.npy')
loss10 = np.load('../result/loss/nonlinear/loss_dt0.1_L20.npy')
loss11 = np.load('../result/loss/nonlinear/loss_dt0.11_L20.npy')
loss13 = np.load('../result/loss/nonlinear/loss_dt0.13_L20.npy')



plt.plot(loss1[:1000], label='lr = 0.01', linewidth=3)
plt.plot(loss3[:1000], label='lr = 0.03', linewidth=3)
plt.plot(loss5[:1000], label='lr = 0.05' ,alpha=0.5)
plt.plot(loss7[:1000], label='lr = 0.07' ,alpha=0.5)
plt.plot(loss9[:1000], label='lr = 0.09' ,alpha=0.5)
plt.plot(loss10[:1000], label='lr = 0.1(stable threshold)')
plt.plot(loss11[:1000], label='lr = 0.11')
plt.plot(loss13[:1000], label='lr = 0.13')


plt.tick_params(labelsize=16)
plt.xlabel('Iteration',fontsize=18, labelpad = -5)
plt.ylabel('Loss',fontsize=18)
plt.legend(loc='upper right', fontsize=14)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.show()