# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:36:45 2019

@author: H.P. Wang
github:  https://github.com/hpwang
"""
import numpy as np
from model import PIV2DSRCNN
import matplotlib.pyplot as plt
import scipy.io as sio



def main():
    """
    use the random data for testing
    """
    test_X = np.random.rand(1,32,32,2) # 1 test data of (32,32,2)
    """
    parameters
    """
    input_size = [32,32]
    sample_size = 1
    saved_name = 'test.h5'
    """
    prediction
    """
    piv2dsr = PIV2DSRCNN(input_size=input_size, input_num=sample_size, 
                        saved_result=saved_name, is_training=False,
                        gamma=0, filter_num=64)
    pred_Y = piv2dsr.process(test_X)
    """
    plot
    """
    plt.figure()
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    ax1.pcolor(test_X[0,:,:,0],cmap='RdYlGn_r', vmin=-1, vmax=1)
    ax1.set_title('PIV data',fontsize=12,color='r')
    ax2.pcolor(pred_Y[0,:,:,0],cmap='RdYlGn_r', vmin=-1, vmax=1) 
    ax2.set_title('Predicted data',fontsize=12,color='r')
    #plt.colorbar()
    plt.show()
                 
if __name__ == '__main__':
    main()
