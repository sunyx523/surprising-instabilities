'''Plots Figure 3 in the paper'''
import numpy as np
import os.path
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

loss1 = np.load('../result/loss/linear/loss_alpha11000000.0.npy')
loss2 = np.load('../result/loss/linear/loss_alpha12000000.0.npy')
loss3 = np.load('../result/loss/linear/loss_alpha13000000.0.npy')
loss4 = np.load('../result/loss/linear/loss_alpha190000000.0.npy')
loss5 = np.load('../result/loss/linear/loss_alpha200000000.0.npy')
loss6 = np.load('../result/loss/linear/loss_alpha210000000.0.npy')
loss7 = np.load('../result/loss/linear/loss_alpha220000000.0.npy')

plt.loglog(loss1[:500], label=r'$\alpha = 1.1e7 < \alpha_{min}$')
plt.loglog(loss2[:500], label=r'$\alpha = 1.2e7 < \alpha_{min}$')
plt.loglog(loss3[:500], label=r'$\alpha = 1.3e7 = \alpha_{min}$')
plt.loglog(loss4[:500], label=r'$\alpha_{min} < \alpha = 1.9e8 < \alpha_{max}$')
plt.loglog(loss5[:500], label=r'$\alpha = 2.0e8 = \alpha_{max}$')
plt.loglog(loss6[:200], label=r'$\alpha = 2.1e8 > \alpha_{max}$')
plt.loglog(loss7[:110], label=r'$\alpha = 2.2e8 > \alpha_{max}$')

plt.legend(fontsize=14)
plt.grid()
plt.xlabel('Iteration',fontsize=18, labelpad = -0.8)
plt.ylabel('Loss',fontsize=18, labelpad = -0.8)
plt.tick_params(labelsize=16)
plt.show()