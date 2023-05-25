'''Code for Figure 3 and Table 2 in Section 5.2'''

import os
import copy
import math
import numpy as np

# Parameters
np.random.seed(7)
num_epochs = 1000
beta = 1
learning_rate = 1e-8 #change for different learning rate
ksize = 32
isize = 256
weight_decay = 2.1e8
sigmoid_shift = 0
address = '../result/loss/'

# Input image
images = np.ones((1, 1, isize, isize))
for m in range(images.shape[2]):
    for n in range(images.shape[3]):
        if (m//1 + n//1) % 2 == 0:
            images[0, 0, m, n] = -1

# Calcuate theoretical alpha_min
images_fft = np.fft.fft2(images)
images_fft_L2 = np.power(images_fft.real, 2) + np.power(images_fft.imag, 2)
print(0.25*np.max(images_fft_L2))  

labels = np.array([1])
loss_list = []
weights = np.random.randn(1, 1, ksize, ksize) #initialize weight and gradients
w_gradient = np.zeros(weights.shape)

for epoch in range(num_epochs):

    # Convolution operation
    input_shape = images.shape
    output_shape = [input_shape[0], 1, input_shape[2] - ksize + 1, input_shape[3] - ksize + 1]
    col_weights = weights.reshape([1 ,-1])   # (c_out, c_in * ksize * ksize)
    col_image = []
    conv_out = np.zeros(output_shape) 
    for i in range(input_shape[0]):
        image = images[i][np.newaxis, :]
        col_image_i = []
        for m in range(0, image.shape[2] - ksize + 1, 1):
            for n in range(0, image.shape[3] - ksize + 1, 1):
                col = image[:, :, (m):(m + ksize), (n):(n + ksize)].reshape([-1])
                col_image_i.append(col)
        col_image_i = np.array(col_image_i).T # (c_in * ksize * ksize, w_out*h_out)
        conv_out[i] = np.reshape(np.dot(col_weights, col_image_i), conv_out[i].shape)   #(c_out, w_out*h_out) -> ( c_out, w_out, h_out)
        col_image.append(col_image_i)
    col_image = np.array(col_image)
    
    swish_out = conv_out * (np.exp(beta * conv_out) / (np.exp(beta * conv_out) + 1)) #swish
    pooling_out = np.mean(np.mean(swish_out, axis=3), axis=2) - sigmoid_shift #pooling
    sigmoid_out = np.exp(pooling_out) / (np.exp(pooling_out) + 1) #sigmoid
    sigmoid_out = sigmoid_out.squeeze()
    loss = np.mean(-1. * labels * np.log(sigmoid_out) - (1 - labels) * np.log(1 - sigmoid_out)) #BCE loss
  
    # Backward propogation
    sigmoid_eta = -0.5 #constant a
    w_gradient = 0.5 * sigmoid_eta * (np.mean(np.mean(images, axis=3), axis=2)  + beta * np.dot(conv_out[0].reshape([1, 1, -1]), col_image[0].T)).reshape(weights.shape)
    w_gradient += weight_decay * weights 
    weights -= learning_rate * w_gradient #update weight
    w_gradient = np.zeros(weights.shape)
            
    L2 = np.square(weights).sum() #regularization loss
    loss_total = loss + weight_decay / 2 * L2
    loss_list.append(loss_total)

    print ('Epoch [{}/{}], Loss: {:.32f}'
                .format(epoch+1, num_epochs, loss_total))

# Save the loss
if not os.path.exists(address):
    os.makedirs(address)
np.save(address + 'loss_alpha'+str(weight_decay)+'.npy', loss_list)







        

