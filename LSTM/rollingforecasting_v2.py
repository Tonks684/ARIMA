# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:21:28 2020

@author: nt385534
"""
import numpy as np
import math

def rolling_forecasting(data, stepsin, numberoffeatures, forecasting_horizon,basemodelnodroupout,scaler):
    
    data1 = data
    data_actual = scaler.inverse_transform(data1)
    data1_test_predict = data1.reshape((1, stepsin, numberoffeatures))
    for i in range(0,forecasting_horizon):
        yhat = basemodelnodroupout.predict(data1_test_predict, verbose=0)
        yhat_actual = scaler.inverse_transform(yhat)
        #Append first predicted value to orig series
        data1 = np.append(data1,yhat[0,:])
        #Take the last "stepsin" of new appended series
        data1_final = data1[-stepsin:]
        #Repeat for actuals
        data_actual = np.append(data_actual,yhat_actual[0,:])
        data_actual_final = data_actual[-stepsin:]
        #Reshape for input back into the model
        data1_test_predict = data1_final.reshape((1, stepsin, numberoffeatures))
        
    data1 = data1[-forecasting_horizon:]
    data_actual = data_actual[-forecasting_horizon:]
    
    return data1, data_actual


def stepout_forecasting(data, stepsin,numberoffeatures, stepsout,forecasting_horizon,basemodelnodroupout,scaler,model_type):
    
    number_of_pred = math.ceil(forecasting_horizon/stepsout)

    #Multistep-Forecast: Appending yhat to input and rerunning model
    if  model_type == 'ms':
        times_stepout = math.ceil(forecasting_horizon/stepsout)
        data1 = data
        data_actual = scaler.inverse_transform(data1)
        data1_test_predict = data1.reshape((1, stepsin, numberoffeatures))
        for i in range(0,times_stepout):
            yhat = basemodelnodroupout.predict(data1_test_predict, verbose=0)
            yhat_actual = scaler.inverse_transform(yhat)
            data1 = np.append(data1,yhat)
            data1_final = data1[-stepsin:]
            data_actual = np.append(data_actual,yhat_actual)
            data_actual_final = data_actual[-stepsin:]
            data1_test_predict = data1_final.reshape((1, stepsin, numberoffeatures))
    
        #print(times_stepout)
        #print(data_actual.shape) 
        #print(data1.shape)
        endingobs = stepsout*times_stepout - forecasting_horizon
        if endingobs != 0:
            data1 = data1[stepsin:-endingobs]
            data_actual = data_actual[stepsin:-endingobs]
        else:
            data1 = data1[stepsin:]
            data_actual = data_actual[stepsin:]	
    
        return data1, data_actual

    #Multi-variate: Input is last stepsin x no.offeatures of train + testset.iloc[:-stepsout,:]
    if model_type == 'mv':
        predictions, predictions_actual = [],[]
        data = np.array(data)
        times_stepout = math.ceil((data.shape[0]-stepsin)/stepsout)
        #print(times_stepout)
        #starting from the first stepsin X no.of features (train_set) continue to add following stepsin
        data1_test_predict = data[:stepsin]
        for i in range(0, times_stepout):
            #reshape into [1, n_input, n] for model
            data1_test_predict = data1_test_predict.reshape((1, data1_test_predict.shape[0], data1_test_predict.shape[1]))
            #Run Model & Adjust shape of yhat
            yhat = basemodelnodroupout.predict(data1_test_predict, verbose=0)
            yhat = yhat.reshape(-1,1)
            #print(yhat)
            yhat_actual = scaler.inverse_transform(yhat)
            yhat_actual = yhat_actual.reshape(-1,1)
            #print(yhat_actual)

            #Append predictions
            if i != times_stepout-1:
                predictions_actual.append(yhat_actual)
                predictions.append(yhat)
            elif times_stepout % stepsout == 0:
                predictions_actual.append(yhat_actual)
                predictions.append(yhat)
            else:
                #Extract those values not needed
                remaining_pred = data.shape[0] - stepsin - (stepsout*times_stepout-1)
                
                #Final Append
                yhat_actual = np.array(yhat_actual)
                predictions_actual.append(yhat_actual[:remaining_pred-1])
                predictions.append(yhat[:remaining_pred-1])
        
            #Adjust input
            data1_test_predict = data[stepsout*(i+1):(stepsin+ ((i+1)*stepsout)), :]
        #Convert to array
        predictions_actual = np.array(predictions_actual)
        predictions = np.array(predictions)
        
        #Extracted Nested Values
        predictions_actual_final = []
        predictions_final = []
        #print("Prediction",predictions_actual.shape)
        
        #Value shoudld be the same as times_stepsout
        for array in range(predictions_actual.shape[0]):
            for index, value in enumerate(predictions_actual[array]):
                predictions_actual_final.append(predictions_actual[array].item(index))
                predictions_final.append(predictions[array].item(index))

        #Convert Back to array
        predictions_actual_final = np.array(predictions_actual_final)
        predictions_actual_final = predictions_actual_final.reshape(-1,1)
        predictions_final = np.array(predictions_final)
        predictions_final = predictions_final.reshape(-1,1)
        
        return predictions_final, predictions_actual_final
                
        #MultiParalell-Forecast: Appending yhat to input and rerunning model
    elif  model_type == 'mp':
        reshape_value = math.ceil(data.shape[0]/stepsin)
        data1_test_predict = data1.reshape((reshape_value, stepsin, numberoffeatures))
        data_actual = scaler.inverse_transform(data1)

        for i in range(0,times_stepout):
            yhat = basemodelnodroupout.predict(data1_test_predict, verbose=0)
            #Reshape yhat from (5,4,15) to (15,5,4)
            yhat = yhat.reshape(yhat.shape[2],yhat.shape[0],yhat.shape[1])
            #Loop through each column prediction
            for y in range(yhat.shape[0]):
                yhat_actual = scaler.inverse_transform(yhat[y])  
                data1 = np.append(data1[:,y],yhat[y])
                data1_final = data1[-stepsin:,:]
                data_actual = np.append(data_actual,yhat_actual)
                data_actual_final = data_actual[-stepsin:,:]
                data1_test_predict = data1_final.reshape((reshape_value, stepsin, numberoffeatures))
            
        endingobs = stepsout*times_stepout - forecasting_horizon
        if endingobs != 0:
            data1 = data1[stepsin:-endingobs]
            data_actual = data_actual[stepsin:-endingobs]
        else:
            data1 = data1[stepsin:]
            data_actual = data_actual[stepsin:]	
    
        return data1, data_actual
        
        
        
        
        return predictions_final, predictions_actual_final