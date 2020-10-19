# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:22:04 2020

@author: smt29021
"""

import pandas as pd
    
def model_type(train,test,timesteps,stepsout,forecasting_horizon,persplit,epoch_num,batch_size,dropout_value):

# define model type (one-to-one, one-to-may, many-to-one, many-to-many)
 # if =1 then one-to-...)    
    if timesteps == 1:
        modelin = 'One-to-'
    else:
        modelin = str(timesteps) + '-to-'

 # if =1 then ...-to-one
    if stepsout == 1:
        modelout = 'One'
    else:
        modelout = str(stepsout)

    
    numberoffeatures = train.shape[1]
    print('Model Type: '+ modelin+modelout)
    print('Forecasting Horizon:',forecasting_horizon)    
    print('Forecasting Evaluation based on:',forecasting_horizon)    
    print('Validation based on Sliding Window')
    print('Partition Node Test set:',persplit)
    print('Train Set Size:', train.shape)
    print('Test Set Size:',test.shape)
    print('Number of features:',train.shape[1])
    print('Epoch:',epoch_num)
    print('Batch size:',batch_size)
    print('Dropout value:',dropout_value)    
    data = [['Model Type',modelin+modelout],['Forecasting Horizon',forecasting_horizon],['Forecasting Evaluation based on',forecasting_horizon],['Partition Node Test set',persplit],['Train Set Size',train.shape],['Test Set Size',test.shape],['Number of features',train.shape[1]],['Epoch',epoch_num],['Batch size',batch_size],['Dropout value',dropout_value]]
    df = pd.DataFrame(data,columns=['Description','Input'])


    return numberoffeatures, df