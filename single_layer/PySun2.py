'''Code for Figure4 right in Section 5.4'''
import os
import copy
import math
import numpy as np
import scipy.io as io
import scipy
np.random.seed(7)

class Conv2D():
    def __init__(self, input_channels, output_channels, ksize=3, stride=1):
        #inilization
        self.input_shape = 0  # input shape (batch_size, w, h, c)
        self.output_channels = output_channels  
        self.input_channels = input_channels 
        self.batchsize = 0  
        self.ksize = ksize  
        self.stride = stride  
        self.weights = np.random.randn(self.output_channels, self.input_channels, ksize, ksize)
        self.eta = 0
        self.w_gradient = np.zeros(self.weights.shape)
        self.buf = np.zeros(self.weights.shape)
        self.output_shape = 0

    def forward(self, x):
        self.input_shape = x.shape
        self.batchsize = self.input_shape[0]
        self.output_shape = [self.batchsize, self.output_channels, int((self.input_shape[2] - self.ksize) / self.stride) + 1, int((self.input_shape[3] - self.ksize) / self.stride) + 1]
        col_weights = self.weights.reshape([self.output_channels ,-1])   # (c_out, c_in * ksize * ksize)

        self.col_image = []
        conv_out = np.zeros(self.output_shape) 

        for i in range(self.batchsize):
            image = x[i][np.newaxis, :]
            self.col_image_i = []
            for m in range(0, image.shape[2] - self.ksize + 1, self.stride):
                for n in range(0, image.shape[3] - self.ksize + 1, self.stride):
                    col = image[:, :, (m):(m + self.ksize), (n):(n + self.ksize)].reshape([-1])
                    self.col_image_i.append(col)
            self.col_image_i = np.array(self.col_image_i).T # (c_in * ksize * ksize, w_out*h_out)
            conv_out[i] = np.reshape(np.dot(col_weights, self.col_image_i), conv_out[0].shape)   #(c_out, w_out*h_out) -> ( c_out, w_out, h_out)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out

    def backward(self, eta):

        self.eta = eta

        col_eta = eta.reshape([self.batchsize, self.output_channels, -1]) #(batch_size, c_out, w_out*h_out)

        for i in range(self.batchsize):
            # dL/dw = dy/dw * dL/dy = x * eta
            self.w_gradient += np.dot(col_eta[i], self.col_image[i].T).reshape(self.weights.shape)  
            # (c_out, w_out*h_out) * (w_out*h_out, c_in * ksize * ksize)  = (c_out, c_in * ksize * ksize) -> (c_out, c_in, ksize, ksize)

        # dL/dx = dL/dy * dy/dx = eta * w
        padding_value = ((self.stride**2 - 1) * self.eta.shape[2] +
                         self.ksize + self.stride *
                         (self.ksize - self.stride - 1)) / 2
        padding_value = math.ceil(padding_value)  
        pad_eta = np.pad(self.eta, ((0, 0), (0, 0), (padding_value, padding_value),
                                    (padding_value, padding_value)), 'constant', constant_values=0)  # zero padding
        flip_weights = np.transpose(np.flipud(np.fliplr(np.transpose(self.weights, (2, 3, 0, 1)))), (3, 2, 0, 1))
        col_flip_weights = flip_weights.reshape([self.input_channels, -1]) # (c_in, c_out*ksize*ksize)

        col_pad_eta = []
        for i in range(self.batchsize):
            col_pad_eta_i = []
            pad_eta_i = pad_eta[i][np.newaxis, :]
            for m in range(0, pad_eta.shape[2] - self.ksize + 1, self.stride):
                for n in range(0, pad_eta.shape[3] - self.ksize + 1, self.stride):
                    col = pad_eta_i[:, :, (m):(m + self.ksize), (n):(n + self.ksize)].reshape([-1])
                    col_pad_eta_i.append(col)
            col_pad_eta_i = np.array(col_pad_eta_i)
            col_pad_eta.append(col_pad_eta_i)
        col_pad_eta = np.array(col_pad_eta)

        next_eta = np.dot(col_pad_eta, col_flip_weights.T) #(batch_size, w_in*h_in, c_out*ksize*ksize) * (c_out*ksize*ksize, c_in) => 
        next_eta = np.reshape(next_eta, self.input_shape)  #(batch_size, w_in, h_in, c_in)
        return next_eta

    def update(self, lr=0.1, weight_decay=0, momentum = 0, nesterov = False, k = 1):
        self.w_gradient += weight_decay * self.weights

        if momentum != 0:
            self.buf = self.buf * momentum + self.w_gradient
            if nesterov:
                self.weights -= (lr * k) / self.batchsize * ((self.w_gradient + momentum * self.buf) / k)
            else:
                self.weights -= (lr * k) / self.batchsize * (self.buf / k)
        else:
            self.weights -= (lr * k) / self.batchsize * (self.w_gradient / k)

        self.w_gradient = np.zeros(self.weights.shape)
    
    def getParameter(self):
        return self.weights


class AvgPooling():
    def __init__(self):
        self.input_shape = 0

    def forward(self, x):
        self.input_shape = x.shape
        out = np.mean(np.mean(x, axis=3), axis=2)
        return out

    def backward(self, eta):
        next_eta = np.expand_dims(eta, axis=2).repeat(self.input_shape[2], axis=2) / self.input_shape[2]
        next_eta = np.expand_dims(next_eta, axis=3).repeat(self.input_shape[3], axis=3) / self.input_shape[3]
        return next_eta

class Swish():
    def __init__(self, b=1):
        self.x = 0
        self.b = b

    def forward(self, x):
        self.x = x
        return self.x * (np.exp(self.b*self.x) / (np.exp(self.b*self.x) + 1))

    def backward(self, eta):
        grad = np.exp(self.b*self.x)/(1+np.exp(self.b*self.x)) + self.x * (self.b*np.exp(self.b*self.x) / ((1+np.exp(self.b*self.x))*(1+np.exp(self.b*self.x))))
        next_eta = eta * grad
        return next_eta

class BCELoss():
    def __init__(self):
        self.x = 0
        self.label = 0
        self.sigmoid = 0

    def forward(self, x, label):
        self.x = x
        self.label = label
        self.sigmoid = np.exp(self.x) / (np.exp(self.x) + 1)
        if self.sigmoid == 1.:
            return np.mean(-1. * self.label * np.log(self.sigmoid))
        elif self.sigmoid == 0.:
            return np.mean(-1. * (1 - self.label) * np.log(1 - self.sigmoid))
        return np.mean(-1. * self.label * np.log(self.sigmoid) - (1 - self.label) * np.log(1 - self.sigmoid))
        

    def backward(self):
        return self.sigmoid - self.label
    
class Model_1LayerCNN():
    def __init__(self, ksize=3):
        self.conv1 = Conv2D(input_channels=1, output_channels=1, ksize=ksize, stride=1)
        self.swish1 = Swish(b=1)
        self.conv2 = Conv2D(input_channels=1, output_channels=1, ksize=ksize, stride=1)
        self.swish2 = Swish(b=1)
        self.avgpool = AvgPooling()
        self.criterion = BCELoss()
        self.loss = 0
        self.out = 0
        self.seq = [
            self.conv1, self.swish1, self.avgpool
        ]

    def forward(self, image):
        self.out = image
        for layer in self.seq:
            self.out = layer.forward(self.out)
        return self.out

    def cal_loss(self, label):
        self.loss = self.criterion.forward(self.out, label)
        return self.loss

    def getA(self):
        return self.criterion.backward()

    def backward(self):
        eta = self.criterion.backward()
        for layer in reversed(self.seq):
            eta = layer.backward(eta)

    def update(self, lr=0.1, weight_decay=0, momentum = 0, nesterov = False, k = 1):
        for layer in reversed(self.seq):
            if 'update' in dir(layer):
                layer.update(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, k=k)

    def getParameter(self):
        parameter = []
        for layer in self.seq:
            if 'getParameter' in dir(layer):
                parameter.append(layer.getParameter())
        return parameter

# Parameters
num_epochs = 1000
learning_rate = 0.05 #change for different learning rate
ksize = 32
isize = 256
weight_decay = 20
address1 = '../parameter/1layer/lr' + str(learning_rate) + '_L' + str(weight_decay) + '/1/'
address2 = '../parameter/1layer/lr' + str(learning_rate) + '_L' + str(weight_decay) + '/2/'
if not os.path.exists(address1):
        os.makedirs(address1)
if not os.path.exists(address2):
    os.makedirs(address2)

# Input image
images = np.ones((1, 1, isize, isize))
for m in range(images.shape[2]):
    for n in range(images.shape[3]):
        if (m//1 + n//1) % 2 == 0:
            images[0, 0, m, n] = -1
    
labels = np.array([1]) #label 1
model1 = Model_1LayerCNN(ksize=ksize)
model2 = copy.deepcopy(model1)

i = 0
for epoch in range(num_epochs):
    model1.forward(images)
    model2.forward(images)
    loss1 = model1.cal_loss(labels)
    loss2 = model2.cal_loss(labels)
    model1.backward()
    model2.backward()
    model1.update(lr = learning_rate, weight_decay = weight_decay, k = 1)
    model2.update(lr = learning_rate, weight_decay = weight_decay, k = 3)
    print ('Epoch [{}/{}], Loss1: {:.32f}, Loss2: {:.32f}'.format(epoch+1, num_epochs, loss1, loss2))        
    
    # Save the loss
    for j, layer in enumerate(model1.getParameter()):
        scipy.io.savemat(address1 + '/' + str(i+1) + '_conv_weight.mat', {'conv_weight': layer})
                
    for j, layer in enumerate(model2.getParameter()):
        scipy.io.savemat(address2 + '/' + str(i+1) + '_conv_weight.mat', {'conv_weight': layer})
        i = i + 1









        

