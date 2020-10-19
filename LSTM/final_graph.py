# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:13:19 2020

@author: smt29021
"""
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.tools as tls

def trainset_graph(train_rescaled,predictions_new,scaler,model_type):
    train_rescaled_actual = scaler.inverse_transform(train_rescaled)
    predictions_new = scaler.inverse_transform(predictions_new) 
    
    #mpl_fig7 = pyplot.figure(figsize=(20,5))
    if model_type == 'mv':
        pyplot.figure(figsize=(50,10))
        train_rescaled_actual = pd.DataFrame(train_rescaled_actual)
        train_rescaled_actual = train_rescaled_actual.iloc[:,-1]
        pyplot.plot(np.array(train_rescaled_actual), color = 'b')
        pyplot.plot(predictions_new,linestyle='--',color='r')
        pyplot.xlabel('Week No.')
        pyplot.ylabel('Actual Sales (rescaled)')
        pyplot.title(' Train set (blue) vs Predictions (red)')
        
        ##Used for WEBAPI - return graph in JSON Format
        #mpl_fig7 = pyplot.figure(figsize=(20,5))
        #pyplot.plot(np.array(train_rescaled_actual), color = 'b')
        #pyplot.plot(predictions_new,linestyle='--',color='r')
        #pyplot.xlabel('Week No.')
        #pyplot.ylabel('Actual Sales (rescaled)')
        #pyplot.title(' Train set (blue) vs Predictions (red)')
        #plotly_fig7=tls.mpl_to_plotly(mpl_fig7)

    elif model_type =='ms':
        pyplot.plot(train_rescaled_actual, color = 'b')
        pyplot.plot(predictions_new,linestyle='--',color='r')
        pyplot.xlabel('Week No.')
        pyplot.ylabel('Actual Sales (rescaled)')
        pyplot.title(' Train set (blue) vs Predictions (red)')
        
        ##Used for WEBAPI - return graph in JSON Format
        #mpl_fig7 = pyplot.figure(figsize=(20,5))
        #pyplot.plot(train_rescaled_actual, color = 'b')
        #pyplot.plot(predictions_new,linestyle='--',color='r')
        #pyplot.xlabel('Week No.')
        #pyplot.ylabel('Actual Sales (rescaled)')
        #pyplot.title(' Train set (blue) vs Predictions (red)')
        #plotly_fig7=tls.mpl_to_plotly(mpl_fig7)
        #plt.subplots_adjust(wspace = 0.4, hspace = 0.4)

    elif model_type == 'mp':
        rows = predictions_new.shape[1]//2 +1
        fig = plt.figure(figsize=(20,45))
        for i in range(predictions_new.shape[1]):
                ax = fig.add_subplot(rows,2,i+1)
                pyplot.plot(train_rescaled_actual[:,i], alpha=0.5, label='Train Set')
                pyplot.plot(predictions_new[:,i], alpha=0.5, label='Predictions')
                pyplot.title(train_rescaled.columns[i])
                pyplot.legend(loc='upper right')   
              
                ##Used for WEBAPI - return graph in JSON Format
                #mpl_fig7 = plt.figure(figsize=(20,45))
                #ax = fig.add_subplot(rows,2,i+1)
                #pyplot.plot(train_rescaled_actual[:,i], alpha=0.5, label='Train Set')
                #pyplot.plot(predictions_new[:,i], alpha=0.5, label='Predictions')
                #pyplot.title(train_rescaled.columns[i])
                #pyplot.legend(loc='upper right')
                #plotly_fig7=tls.mpl_to_plotly(mpl_fig7)
                 
 
    return train_rescaled_actual,predictions_new, #plotly_fig7

def testset_graph(test_rescaled,predictions_new,scaler,timeseriesname,model_type):
    
    test_rescaled_actual = scaler.inverse_transform(test_rescaled)
    predictions_new = scaler.inverse_transform(predictions_new) 
    pyplot.figure()
    #mpl_fig7 = pyplot.figure(figsize=(20,5))
    
    if model_type =='mv':
    
        test_rescaled_actual = pd.DataFrame(test_rescaled_actual)
        test_rescaled_actual = test_rescaled_actual.iloc[:,-1]
        pyplot.plot(np.array(test_rescaled_actual), color = 'b')
        pyplot.plot(predictions_new,linestyle='--',color='r')
        pyplot.xlabel('Week No.')
        pyplot.ylabel('Actual Sales (rescaled)')
        pyplot.title(' Test set (blue) vs Predictions (red)')
       
        #Json format
        #pyplot.plot(np.array(test_rescaled_actual), color = 'b')
        ##mpl_fig7 = pyplot.figure(figsize=(20,5))
        #pyplot.plot(predictions_new,linestyle='--',color='r')
        #pyplot.xlabel('Week No.')
        #pyplot.ylabel('Actual Sales (rescaled)')
        #pyplot.title(' Test set (blue) vs Predictions (red)')
        #plotly_fig7=tls.mpl_to_plotly(mpl_fig7)
    
    elif model_type == 'ms':

        test_rescaled = test_rescaled.reshape(-1,1)
        pyplot.plot(test_rescaled_actual, color = 'b')
        pyplot.plot(predictions_new[:len(test_rescaled_actual)],linestyle='--',color='r')
        pyplot.xlabel('Week No.')
        pyplot.ylabel('Actual Sales (rescaled)')
        pyplot.title(' Test set (blue) vs Predictions (red)')
        #Json format
        #mpl_fig7 = pyplot.figure(figsize=(20,5))
        #pyplot.plot(test_rescaled_actual, color = 'b')
        #pyplot.plot(predictions_new,linestyle='--',color='r')
        #pyplot.xlabel('Week No.')
        #pyplot.ylabel('Actual Sales (rescaled)')
        #pyplot.title(' Test set (blue) vs Predictions (red)')
        #plotly_fig7=tls.mpl_to_plotly(mpl_fig7)
        
        predictions_new = predictions_new[:len(test_rescaled_actual)]
        
    elif model_type == 'mp':
        rows = predictions_new.shape[1]//2 +1
        fig = plt.figure(figsize=(20,45))
        for i in range(predictions_new.shape[1]):
                ax = fig.add_subplot(rows,2,i+1)
                pyplot.plot(test_rescaled_actual[:,i], alpha=0.5, label='Test Set')
                pyplot.plot(predictions_new[:,i], alpha=0.5, label='Predictions')
                pyplot.title(test_rescaled.columns[i])
                pyplot.legend(loc='upper right')   
              
                #Json format
                #mpl_fig7 = plt.figure(figsize=(20,45))
                #ax = fig.add_subplot(rows,2,i+1)
                #pyplot.plot(train_rescaled_actual[:,i], alpha=0.5, label='Train Set')
                #pyplot.plot(predictions_new[:,i], alpha=0.5, label='Predictions')
                #pyplot.title(test_rescaled.columns[i])
                #pyplot.legend(loc='upper right')
                #plotly_fig7=tls.mpl_to_plotly(mpl_fig7)

    filenamefinal = timeseriesname + '_test_predictions.csv'    
    pd.DataFrame(predictions_new).to_csv(filenamefinal)
    
    return test_rescaled_actual,predictions_new #,plotly_fig7
              



def final_graph(OriginalDataSeries,predictions_forecasting,total_scaler,timeseriesname):
    
    predictions_forecasting = total_scaler.inverse_transform(predictions_forecasting)
    forecasting_horizon = predictions_forecasting.shape[0]
    
    #Set to zero to work with floats
    forecasting_set_final = np.zeros((OriginalDataSeries.shape[0]+forecasting_horizon,1))
        
    #Create each series with base nan values
    forecasting_set_final[0:] = np.nan

    #Update each series with required values
    forecasting_set_final[-forecasting_horizon:] = pd.DataFrame(predictions_forecasting)
    
    if OriginalDataSeries.shape[1] == 1:
    
    #Plot
        pyplot.figure(figsize=(20,5))
        pyplot.plot(OriginalDataSeries,label='Sales')
        pyplot.plot(forecasting_set_final, linestyle='--',color = 'k',label='Predictions')
        pyplot.xlabel('Date')
        pyplot.ylabel('Sales')
        pyplot.legend()
        pyplot.show
    
    # Json Plot
        #mpl_fig7 = pyplot.figure(figsize=(20,5))
        #pyplot.plot(OriginalDataSeries,label='Sales')
        #pyplot.plot(forecasting_set_final, linestyle='--',color = 'k',label='Predictions')
        #pyplot.xlabel('Date')
        #pyplot.ylabel('Sales')
        #pyplot.legend()
        #plotly_fig7=tls.mpl_to_plotly(mpl_fig7)
    
    #Final data with prediction
        final_data_prediction = np.concatenate((OriginalDataSeries,predictions_forecasting[-forecasting_horizon:]))
        final_data_prediction = pd.DataFrame(final_data_prediction)
       
    else:
    
    #Plot
        pyplot.figure(figsize=(20,5))
        pyplot.plot(OriginalDataSeries[:,-1],label='Sales')
        pyplot.plot(forecasting_set_final, linestyle='--',color = 'k',label='Predictions')
        pyplot.xlabel('Date')
        pyplot.ylabel('Sales')
        pyplot.legend()
        pyplot.show
    
    #Json Plot
        #mpl_fig7 = pyplot.figure(figsize=(20,5))
        #pyplot.plot(OriginalDataSeries[:,-1],label='Sales')
        #pyplot.plot(forecasting_set_final, linestyle='--',color = 'k',label='Predictions')
        #pyplot.xlabel('Date')
        #pyplot.ylabel('Sales')
        #pyplot.legend()
        #plotly_fig7=tls.mpl_to_plotly(mpl_fig7)
        
    #Final data with prediction
        OriginalDataSeries_final = OriginalDataSeries[:,-1]
        OriginalDataSeries_final = OriginalDataSeries_final.reshape(-1,1)
        print(OriginalDataSeries_final.shape)
        print(predictions_forecasting.shape)
        final_data_prediction = np.concatenate((OriginalDataSeries_final,predictions_forecasting[-forecasting_horizon:]))
        final_data_prediction = pd.DataFrame(final_data_prediction)
        
    filenamefinal = timeseriesname + '_final_predictions.csv'    
    final_data_prediction.to_csv(filenamefinal)
    
    return final_data_prediction#, plotly_fig7