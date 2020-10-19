# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:13:19 2020

@author: smt29021
"""

import numpy as np
import pandas as pd

def predictions_new(train,predictions, stepsin,stepsout,model_type):

    if model_type == 'ms':
        predictions_new = np.zeros((len(train)+(stepsout+stepsin-1),1))
        predictions_new[0:stepsin,0] = np.nan
        for i in range(1,predictions.shape[0]):
            predictions_new[stepsin-1+i,:] = predictions[i-1,0] 
            predictions_new[-stepsout:,0] = predictions[predictions.shape[0]-1,:]  
    
    elif model_type == 'mv':
        predictions_new = np.zeros((len(train)+(stepsout+stepsin-2),1))
        predictions_new[0:stepsin,0] = np.nan
        for i in range(1,predictions.shape[0]):
            predictions_new[stepsin-1+i,:] = predictions[i-1,0] 
            predictions_new[-stepsout:,0] = predictions[predictions.shape[0]-1,:]  
    
    elif model_type == 'mp':
        # Reshaping array into number of series x length of series x stepsout
        # 117x8x15 -> 15x117x8
        predictions = predictions.reshape(predictions.shape[2],predictions.shape[0],predictions.shape[1])
        print('predictions',predictions.shape)
        # Create New Predictions Matrix shape length of series x number of series
        predictions_new = np.zeros((predictions.shape[0],predictions.shape[1]+(stepsout+stepsin-1),predictions.shape[2]))
        print('predictions_new',predictions_new.shape)
        # NAN for first stepsin number of rows and number of series
        predictions_new[:,0:stepsin,:] = np.nan
        
        # Loop through each paralell series
        for i in range(predictions.shape[0]-1):
        # Append first prediction of each of 117 predictions to New Predictions array
            for j in range(predictions.shape[1]):
                predictions_new[i,stepsin+j,:] = predictions[i,j,:]
                predictions_new[i,-stepsout:,:] = predictions[i,predictions.shape[1]-1,:]
                
        # Final prediction must be the full 8 for all 15 series
        #predictions_new = predictions_new.reshape(predictions_new.shape[1],predictions_new.shape[0],predictions_new.shape[2])
        #predictions_new[-stepsout:] = predictions[:,predictions.shape[0]-1]
        #predictions_new = predictions_new.reshape(predictions_new.shape[1],predictions_new.shape[0],predictions_new.shape[2])
       
        #print((predictions_new[:,-stepsout:]).shape)
        #print(predictions[:,predictions.shape[0]-1,:])
        #print(np.transpose(predictions[:,predictions.shape[0]-1,:]))
        #print(predictions[:,predictions.shape[0]-1,:].shape)
        
     #   print(predictions[:,predictions.shape[0]-1,:].shape)
        #print(np.transpose(predictions[:,predictions.shape[0]-1]))
        #predictions = predictions.reshape(-1)
        #print((predictions[:,predictions.shape[0]-1]))
        #print(predictions_new[-stepsout:,:])
    return predictions_new