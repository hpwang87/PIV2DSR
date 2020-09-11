# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:09:41 2019

@author: H.P. Wang
github:  https://github.com/hpwang
"""

import numpy as np
from model import PIV2DSRCNN
import matplotlib.pyplot as plt
import scipy.io as sio

  
if __name__ == '__main__':
    """
    use the random data instead of loading the velocity fields
    """
    train_X = np.random.rand(256,16,16,2)     # 256 input data of (16,16,2)
    train_Y = np.random.rand(256,64,64,2)      # the ground-truth of (64,64,2)
    """
    parameters
    """
    sample_size  = 256
    batch_size = 16
    input_size = [16,16]  
    filter_num = 64
    gamma = 0
    epochs = 100
    saved_name = 'test.h5'
    """
    training
    """
    piv2dsr = PIV2DSRCNN(input_size=input_size, input_num=sample_size, saved_result=saved_name, 
                is_training=True, learning_rate=10e-5, gamma=gamma, filter_num=filter_num, 
                batch_size=batch_size, epochs=epochs)
    H = piv2dsr.train(train_X, train_Y)
    """
    save weights and loss
    """
    saved_name = 'test.mat'  # the filename of loss 
    loss_file = './weight/'+ saved_name
    sio.savemat(loss_file, {'train_loss': H.history['loss'], 'val_loss': H.history['val_loss']}) 
    """
    plot
    """
    plt.figure()
    N = np.arange(0,len(H.history['loss']))
    plt.plot(N,H.history['loss'],label='train_loss')
    plt.scatter(N,H.history['loss'])
    plt.plot(N,H.history['val_loss'],label='val_loss')
    plt.scatter(N,H.history['val_loss'])
    plt.title('Training Loss on Our_dataset')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend()
    
