# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:10:01 2019

@author: H.P. Wang
github:  https://github.com/hpwang
"""

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from PIV2DSR import generator
import os
import scipy.io as sio



"""
PIV2DSR
"""
class PIV2DSRCNN():
    def __init__(self, input_size, input_num, saved_result, is_training=False, 
                 learning_rate=1e-4, gamma=0, filter_num=32, batch_size=16, epochs=100):
        self.input_size = input_size    # the input size
        self.input_num = input_num      # the total number of input data
        self.result = saved_result      # the path for weight saving
        self.learning_rate = learning_rate   # learning rate
        self.batch_size = batch_size         # batch size
        self.gamma = gamma                   # the weight for L2 regularization
        self.filter_num = filter_num         # the number of filters (channels or features)
        self.epochs = epochs                 # the number of epochs
        self.is_training = is_training       # true for training, false for testing
        if self.is_training:
            self.model = self.build_model()
        else:
            self.model = self.load()
            
        reduce_lr = LearningRateScheduler(self.constant_scheduler,verbose=1)
        self.call_back_list =[
                ModelCheckpoint(filepath='./weight/' + self.result, 
                                monitor='loss', save_best_only=True, period=10), reduce_lr]
    
    
    def build_model(self):
        """
        bulid the model
        """
        shape = (self.input_size[0], self.input_size[1], 2) 
        model = generator(input_shape=shape,filters=self.filter_num,n_sub_block=2, gamma=self.gamma)
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        model.summary()
     
        return model
    

    def loss(self, y_true, y_pred):
        '''
        user-defined loss function
        '''  
        def _sum_square(x):
            return K.sum(K.square(x), axis=-1)
        loss2 = K.mean(K.square(_sum_square(y_pred) - _sum_square(y_true)), axis=-1)
        
        return loss2
        
    
    
    def train(self, train_X, train_Y):
        """
        train the network
        """
        history = self.model.fit(train_X, train_Y, batch_size=self.batch_size, epochs=self.epochs, 
                                 verbose=1, callbacks=self.call_back_list, validation_split=0.1,
                                 shuffle=True)
        if self.is_training:
            self.save()
        
        return history
    
        
    def process(self, input_X):
        """
        predict the data (input_X)
        """
        predicted = self.model.predict(input_X)
        
        return predicted
    
    
    def load(self):
        """
        load the weight of the network
        """
        weight_file = './weight/'+ self.result
        model = self.build_model()
        model.load_weights(weight_file)
        
        return model
     
    
    def save(self):
        """
        save the weights
        """
        self.model.save_weights('./weight/'+self.result)
        
    

    def piecewise_scheduler(self, epoch):
        """
        piecewise learning rate decay
        """
        rate = np.floor(epoch/100)
        return self.learning_rate/(rate+1.0)
    
    

    def exponential_scheduler(self, epoch):
        """
        exponential learning rate decay
        """
        decay_rate = 0.95
        decay_epoch = 30
        return self.learning_rate * np.power(decay_rate,(epoch / decay_epoch))
        
    

    def constant_scheduler(self, epoch):
        """
        constant learning rate decay
        """
        return self.learning_rate     

