'''Plots Figure 1 in section 3'''
import numpy as np
import matplotlib.pyplot as plt

# Initialization
u0 = np.zeros(31)
for i in range(15):
    u0[i] = i / 10
for i in range(15, 31):
    u0[i] = (30 - i) / 10

niter = 100001
dt = 0.4 # 0.4 (stable) | 0.8 (unstable)
u = u0
for n in range(niter):
    uxx = np.zeros(31)
    uxx[1:-1] = u[2:] + u[:-2] - 2 * u[1:-1]
    uxx[0] = 0
    uxx[-1] = 0
    u = u + dt * uxx
    if n == int(0.8/dt):
        u1 = u
    elif n == int(2.4/dt):
        u2 = u
    elif n == int(4/dt):
        u3 = u
    elif n == int(100/dt) and dt < 0.5:
        u4 = u
    elif n == int(200/dt) and dt < 0.5:
        u5 = u
    elif n == int(800/dt) and dt < 0.5:
        u6 = u

plt.plot(u0, label='t = 0')
plt.plot(u1, '--', label='t = 0.8')
plt.plot(u2, '--', label='t = 2.4')
plt.plot(u3, '--', label='t = 4')
if dt < 0.5:
    plt.plot(u4, '--', label='t = 40')
    plt.plot(u5, '--', label='t = 200')
    plt.plot(u6, '--', label='t = 800')

plt.legend(loc='upper left',fontsize=14)

plt.xlabel('x',fontsize=18, labelpad = -0.5)
plt.ylabel('u',fontsize=18)
plt.grid()
plt.title('')
plt.tick_params(labelsize=16)
plt.ylim([-0.15, 2.45])
plt.show()





