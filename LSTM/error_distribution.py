# -*- coding: utf-8 -*-
"""
Error Distribution

Inputs
    - data: train or test set as array
    - predictions
    - stepsout
    - model_type: either 'ms', 'mv' or 'mp'
    - data_columns: used to populate titles for Multi-parrelell error plots
    
Output
    - Histogram of error distribution
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt 
import pandas as pd 
from matplotlib import pyplot
import matplotlib.pyplot as plt

 
def error_distribution(data,predictions,stepsout,stepsin,model_type,data_columns):
    data = np.array(data)
    if model_type == 'mp':
        rows = predictions.shape[1]//2 +1
        fig = plt.figure(figsize=(20,45))
        
        for i in range(predictions.shape[1]):
                errors = data[stepsout:,i] - predictions[stepsout:,i]
                ax = fig.add_subplot(rows,2,i+1)
                pyplot.hist(errors, alpha=0.5, label='Errors')
                pyplot.title(data_columns.columns[i])
                pyplot.xlabel('Bin range')
                pyplot.ylabel("Error Distribution")
               
        #Need to add in MAE, RMSE, MAPE for each error[i]
    
    elif model_type =='mv':
        x = data[stepsin:]
        y = predictions[stepsin:,0]
        
        # MAE
        errors = x - y
        mae = sum(abs(errors))/len(x)
        print("MAE:%f" % mae)

        # MAPE 
        mape = (1/len(x))* sum(abs((x - y)/x)) * 100
        mape = mape
        print("MAPE:%f" % mape)
    
        # RMSE
        mse = sum(np.square(errors)) * (1/len(x))
        rmse = sqrt(mse)
        print('RMSE:%f' % rmse)
    
        plt.hist(errors)
        plt.ylabel("Error Distribution")
        plt.xlabel('Bin range')
        plt.show()
        
        
    
    else:
        data = data[stepsout:]
        data=pd.DataFrame(data)
        predictions = predictions[stepsout:]
        

        # MAE
        errors = np.array(data - predictions)
        mae = sum(abs(errors))/len(data)
        mae = np.around(mae[0],decimals = 0)
        print("MAE: £%f" % mae)
     
        # MAPE
        mape = sum(abs(np.array((data - predictions)/data)))/len(data)
        mape = mape
        mape = np.around(mape,decimals=2)
        print("MAPE: %f" % mape)
    
        # RMSE
        mse = sum(np.square(errors))
        rmse = sqrt(mse)
        rmse = np.around(rmse,decimals=0)
        print('RMSE: £%f' % rmse)
    
        plt.hist(errors)
        plt.ylabel("Error Distribution")
        plt.xlabel('Bin range')
        plt.show()
  
        
    